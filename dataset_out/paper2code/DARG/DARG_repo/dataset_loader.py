## dataset_loader.py
import os
import json
import logging
from typing import List, Optional
from functools import partial
from utils import load_config

# Configure logger for debugging and info
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataPoint:
    """
    Data class representing a single data entry, including question, answer,
    reasoning explanation, and reasoning graph.
    """
    def __init__(
        self,
        question_text: str,
        answer: str,
        reasoning_text: str,
        reasoning_graph: dict  # The graph stored as a dict (serializable)
    ):
        self.question_text = question_text
        self.answer = answer
        self.reasoning_text = reasoning_text
        self.reasoning_graph = reasoning_graph

    def to_json(self) -> dict:
        """
        Serialize DataPoint into a JSON-serializable dict.
        """
        return {
            'question': self.question_text,
            'answer': self.answer,
            'reasoning': self.reasoning_text,
            'reasoning_graph': self.reasoning_graph
        }

    @staticmethod
    def from_json(data: dict) -> 'DataPoint':
        """
        Create DataPoint instance from dict, validating presence of fields.
        """
        question = data.get('question')
        answer = data.get('answer')
        reasoning = data.get('reasoning')
        reasoning_graph = data.get('reasoning_graph', {})

        if question is None or answer is None or reasoning is None:
            raise ValueError(f"Missing one of required fields in data: {data}")
        return DataPoint(
            question_text=question,
            answer=answer,
            reasoning_text=reasoning,
            reasoning_graph=reasoning_graph
        )

class DatasetLoader:
    """
    Loader class for datasets like GSM8K, BBQ, BBH Dyck.
    Handles loading, parsing, and saving datasets.
    """

    def __init__(self, dataset_path: str, dataset_name: Optional[str] = None):
        """
        Initialize with dataset file path.
        Args:
            dataset_path: Path to dataset JSON file.
            dataset_name: Optional, label for dataset type (e.g., 'GSM8K'), for validation.
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.data_points: List[DataPoint] = []

    def load_data(self) -> List[DataPoint]:
        """
        Load dataset file into self.data_points as DataPoint objects.
        Supports JSON format with list of entries.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        logger.info(f"Loading dataset from {self.dataset_path}")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in {self.dataset_path}: {e}")

        # Support datasets stored as list of dicts
        if not isinstance(raw_data, list):
            raise ValueError(f"Dataset should be a list of data points: {self.dataset_path}")

        self.data_points = []
        for idx, entry in enumerate(raw_data):
            # Validate structure
            if not isinstance(entry, dict):
                logger.warning(f"Skipping non-dict entry at index {idx}")
                continue
            try:
                datapoint = DataPoint.from_json(entry)
            except Exception as e:
                logger.warning(f"Skipping invalid data at index {idx}: {e}")
                continue
            self.data_points.append(datapoint)

        logger.info(f"Loaded {len(self.data_points)} data points from {self.dataset_path}")
        return self.data_points

    def save_dataset(self, output_path: str, data_points: Optional[List[DataPoint]] = None) -> None:
        """
        Save current data points to a JSON file at output_path.
        Args:
            output_path: Path to save the dataset.
            data_points: Optional list of DataPoint objects to save. Defaults to internal.
        """
        to_save = data_points if data_points is not None else self.data_points
        json_list = [dp.to_json() for dp in to_save]

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_list, f, indent=2)
        logger.info(f"Saved {len(json_list)} data points to {output_path}")

    def get_subset(self, sample_size: int, seed: int = 42) -> List[DataPoint]:
        """
        Randomly sample a subset of data points for evaluation or analysis.
        Args:
            sample_size: Number of data points to sample.
            seed: Random seed for reproducibility.
        Returns:
            List of sampled DataPoint objects.
        """
        import random
        random.seed(seed)
        if sample_size >= len(self.data_points):
            return self.data_points
        return random.sample(self.data_points, sample_size)
