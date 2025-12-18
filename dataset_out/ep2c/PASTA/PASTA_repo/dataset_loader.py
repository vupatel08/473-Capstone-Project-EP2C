## dataset_loader.py
import os
import json
import random
from typing import List, Dict, Tuple, Optional
from datasets import Dataset

from utils import load_config, validate_path

class DatasetLoader:
    """
    A class to load, parse, and prepare datasets for different tasks as per the experimental setup.
    Supports BiasBios, CounterFact, JSON Formatting, and Pronouns Changing datasets.
    """
    def __init__(
        self,
        dataset_paths: Dict[str, str],
        task_name: str,
        split_ratios: Dict[str, float] = None,
        seed: int = 42,
        cache_dir: str = "cached_datasets"
    ):
        """
        Initialize DatasetLoader with dataset paths and task info.
        Args:
            dataset_paths (dict): dictionary with dataset paths keyed by dataset name.
            task_name (str): which dataset to load.
            split_ratios (dict): ratios for train/val/test splits, default: {'train':0.6,'val':0.2,'test':0.2}
            seed (int): seed for splitting.
            cache_dir (str): directory to cache processed datasets.
        """
        self.dataset_paths = dataset_paths
        self.task_name = task_name
        self.seed = seed
        self.cache_dir = cache_dir
        if split_ratios is None:
            self.split_ratios = {'train': 0.6, 'val': 0.2, 'test': 0.2}
        else:
            self.split_ratios = split_ratios
        self.dataset = None  # to hold the loaded dataset

        validate_path(cache_dir, must_exist=False)

    def load_dataset(self):
        """
        Load, parse, and split dataset according to task_name.
        Supports caching for efficiency.
        """
        cache_path = os.path.join(self.cache_dir, f"{self.task_name}_full.json")
        if os.path.exists(cache_path):
            # Load preprocessed dataset from cache
            with open(cache_path, 'r') as f:
                self.dataset = json.load(f)
            return

        # Select parsing method based on task name
        if self.task_name == 'BiasBios':
            raw_data = self._load_raw_data('bias_bios')
            parsed_data = self._parse_bias_bios(raw_data)
        elif self.task_name == 'CounterFact':
            raw_data = self._load_raw_data('counterfact')
            parsed_data = self._parse_counterfact(raw_data)
        elif self.task_name == 'JSON Formatting':
            raw_data = self._load_raw_data('json_format')
            parsed_data = self._parse_json_format(raw_data)
        elif self.task_name == 'Pronouns Changing':
            raw_data = self._load_raw_data('pronouns_changing')
            parsed_data = self._parse_pronouns_changing(raw_data)
        else:
            raise ValueError(f"Unknown task name: {self.task_name}")

        # Save full dataset cache
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(parsed_data, f)

        # Split into train/val/test
        self.dataset = self._split_dataset(parsed_data)
        
    def get_dataset(self) -> Dict[str, List[Dict]]:
        """
        Return dict with 'train', 'validation', 'test' datasets.
        """
        if self.dataset is None:
            self.load_dataset()
        return self.dataset

    def _load_raw_data(self, dataset_key: str) -> List[dict]:
        """
        Load raw data file(s) for specified dataset key from path.
        Currently supports JSONL or JSON lines.
        """
        path = self.dataset_paths.get(dataset_key)
        if path is None:
            raise ValueError(f"Dataset path for {dataset_key} not provided")
        validate_path(path)
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    data.append(sample)
                except json.JSONDecodeError:
                    # If dataset is in raw text, extend parsing as needed
                    # For simplicity, assume JSON Lines
                    raise
        return data

    def _split_dataset(self, data: List[dict]) -> Dict[str, List[dict]]:
        """
        Split the dataset into train/val/test according to ratios, fixed seed.
        """
        random.seed(self.seed)
        data_shuffled = data.copy()
        random.shuffle(data_shuffled)
        total = len(data_shuffled)
        train_end = int(total * self.split_ratios['train'])
        val_end = train_end + int(total * self.split_ratios['val'])
        train_data = data_shuffled[:train_end]
        val_data = data_shuffled[train_end:val_end]
        test_data = data_shuffled[val_end:]
        return {'train': train_data, 'validation': val_data, 'test': test_data}

    def _parse_bias_bios(self, raw_data: List[dict]) -> List[dict]:
        """
        Parse BiasBios dataset.
        Expect each sample to have biographical context and occupation label.
        """
        parsed = []
        for sample in raw_data:
            context = sample.get('text', '').strip()
            occupation = sample.get('label', '').strip()
            # Assume emphasis markers are already embedded if provided
            input_text = context
            target_text = occupation
            # Typically, emphasis is on the first sentence; no further parsing needed
            parsed.append({
                'input_text': input_text,
                'target_text': target_text,
                'highlighted_spans': self._extract_emphasis_indices(input_text),
                'task_type': 'BiasBios'
            })
        return parsed

    def _parse_counterfact(self, raw_data: List[dict]) -> List[dict]:
        """
        Parse CounterFact dataset.
        Each sample contains old and new facts, and a question.
        """
        parsed = []
        for sample in raw_data:
            old_fact = sample.get('old_fact', '').strip()
            new_fact = sample.get('new_fact', '').strip()
            question = sample.get('question', '').strip()
            # Input prompt: "Previously, {old_fact}. Currently, {new_fact}. {question}"
            input_text = f"Previously, {old_fact}. Currently, {new_fact}. {question}"
            target_text = new_fact  # Expected output
            # No emphasis marking assuming user provides highlighted spans
            parsed.append({
                'input_text': input_text,
                'target_text': target_text,
                'highlighted_spans': self._extract_emphasis_indices(input_text),
                'task_type': 'CounterFact'
            })
        return parsed

    def _parse_json_format(self, raw_data: List[dict]) -> List[dict]:
        """
        Parse JSON formatting task.
        Each sample contains 'name' and 'occupation'.
        Generate input prompt instructing to produce JSON output.
        """
        parsed = []
        for sample in raw_data:
            name = sample.get('name', '').strip()
            occupation = sample.get('occupation', '').strip()
            # Generate input: "Winnie is an American photographer... {instruction}"
            # Using prompt template in utils or simply embedded here
            input_text = (
                f"{name} is an American {occupation} living in New York. "
                f"Specialized in fashion photography and portrait, she applies her talent on "
                f"both humans and animals. {self._get_instruction('json_format')}"
            )
            # Expected output: JSON object string with name and occupation
            target_text = json.dumps({"name": name, "occupation": occupation})
            parsed.append({
                'input_text': input_text,
                'target_text': target_text,
                'highlighted_spans': self._extract_emphasis_indices(input_text),
                'task_type': 'JSON Formatting'
            })
        return parsed

    def _parse_pronouns_changing(self, raw_data: List[dict]) -> List[dict]:
        """
        Parse Pronouns Changing dataset.
        For each sample, generate prompt with emphasis on context.
        """
        parsed = []
        for sample in raw_data:
            context = sample.get('context', '').strip()
            person_name = sample.get('person', '').strip()
            occupation = sample.get('occupation', '').strip()
            # Assume the emphasis marker is around the context
            input_text = (
                f"{context} You should change 'she' and 'he' to 'they' and generate "
                f"the occupation of {person_name} after changing pronouns."
            )
            target_text = sample.get('target_text', '').strip()
            parsed.append({
                'input_text': input_text,
                'target_text': target_text,
                'highlighted_spans': self._extract_emphasis_indices(input_text),
                'task_type': 'Pronouns Changing'
            })
        return parsed

    def _extract_emphasis_indices(self, text: str, emphasis_marker: str = "**") -> List[int]:
        """
        Extract token indices/character positions of emphasized spans.
        Assumes emphasis markers are embedded with asterisks or similar.
        For simplicity, return character indices of emphasized parts.
        """
        pattern = re.escape(emphasis_marker) + '(.*?)' + re.escape(emphasis_marker)
        spans = [match.span() for match in re.finditer(pattern, text)]
        # Remove markers from text
        clean_text = re.sub(pattern, r'\1', text)
        emphasized_positions = []
        for start, end in spans:
            # Map to character indices of emphasized spans
            emphasized_positions.extend(range(start, end - 2*len(emphasis_marker)))
        return emphasized_positions

    # Utility method to get pre-defined instruction snippets, if needed
    def _get_instruction(self, task_type: str) -> str:
        """
        Retrieve task-specific instruction snippets for prompting.
        """
        # Could be implemented to load from config or hardcoded
        if task_type == 'json_format':
            # Example instruction snippet
            return "Answer the occupation of {person} and generate the answer as json format."
        else:
            return ""

