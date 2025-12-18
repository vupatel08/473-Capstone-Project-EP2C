## dataset_loader.py
import json
import os
import random
from typing import List, Tuple, Dict, Optional

import datasets  # Hugging Face datasets library


class DataEntry:
    """
    Data structure for each dataset sample.
    """
    def __init__(
        self,
        prompt: str,
        response: str,
        class_label: str,
        reward: float
    ):
        self.prompt = prompt
        self.response = response
        self.class_label = class_label
        self.reward = reward

    def to_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "class_label": self.class_label,
            "reward": self.reward
        }


class DatasetLoader:
    """
    Loads and prepares mixed-quality conversation datasets for training and evaluation.
    """
    def __init__(self, config: Dict):
        self.dataset_path: str = config.get("dataset_path", "data/sharegpt_mixed_quality.json")
        self.train_sample_size: int = config.get("train_sample_size", 128)
        self.eval_sample_size: int = config.get("eval_sample_size", 128)
        self.seed: int = config.get("seed", 42)
        self.alpha: float = config.get("alpha", 0.8)  # Reward for sub-optimal data
        self.conditioning_token: str = config.get("conditioning_token", "<|class|>")
        self.data: List[Dict] = []
        self.train_data: List[DataEntry] = []
        self.eval_data: List[DataEntry] = []

        random.seed(self.seed)

        # Load raw dataset
        self._load_raw_dataset()

        # Parse and assign class labels and rewards
        self._parse_and_assign_labels()

        # Sample datasets for train and eval
        self._sample_datasets()

    def _load_raw_dataset(self):
        """
        Loads dataset JSON file into self.data.
        Assumes each line is a JSON object or a JSON array.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            # Try to load as a JSON list
            try:
                raw = json.load(f)
                if isinstance(raw, list):
                    self.data = raw
                else:
                    # If not list, assume JSON object with key 'conversations'
                    self.data = raw.get("conversations", [])
            except json.JSONDecodeError:
                # Fallback: read line by line if dataset is newline-delimited JSON
                self.data = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue

    def _parse_and_assign_labels(self):
        """
        Parses dataset entries, assigns class labels based on source info,
        and computes reward weight.
        """
        parsed_data: List[DataEntry] = []

        for entry in self.data:
            # Expect each entry to have 'prompt', 'response', optional 'source'
            prompt_str = entry.get("prompt", "").strip()
            response_str = entry.get("response", "").strip()
            source = entry.get("source", "").lower()  # e.g., 'gpt-4', 'gpt-3.5'

            # Determine class_label based on source
            if "gpt-4" in source:
                class_label = "expert"
                reward = 1.0
            elif "gpt-3.5" in source:
                class_label = "sub_optimal"
                reward = self.alpha
            else:
                # Default fallback: treat as sub-optimal if source info missing
                class_label = "sub_optimal"
                reward = self.alpha

            # Optionally, incorporate class conditioning token in prompt
            conditioned_prompt = self._apply_class_conditioning(prompt_str, class_label)

            parsed_entry = DataEntry(
                prompt=conditioned_prompt,
                response=response_str,
                class_label=class_label,
                reward=reward
            )
            parsed_data.append(parsed_entry)

        # Shuffle data for randomness
        random.shuffle(parsed_data)

        self.data = parsed_data

    def _apply_class_conditioning(self, prompt: str, class_label: str) -> str:
        """
        Incorporate class conditioning token or prefix into the prompt.
        Format can be customized; here, prepend class token.
        E.g., "<|class|> GPT-4 User: " or "User:" depending on class.
        """
        prefix = ""
        if class_label == "expert":
            prefix = f"{self.conditioning_token} GPT4 User: "
        elif class_label == "sub_optimal":
            prefix = f"{self.conditioning_token} GPT3 User: "
        else:
            prefix = "User: "

        # Append the original prompt to the prefix
        return f"{prefix}{prompt}"

    def _sample_datasets(self):
        """
        Randomly sample training and evaluation datasets based on sample sizes.
        """
        total_data = self.data

        # For reproducibility
        random.seed(self.seed)

        # Sample training data
        train_samples = min(self.train_sample_size, len(total_data))
        self.train_data = random.sample(total_data, train_samples)

        # Sample evaluation data
        remaining_data = [d for d in total_data if d not in self.train_data]
        eval_samples = min(self.eval_sample_size, len(remaining_data))
        self.eval_data = random.sample(remaining_data, eval_samples)

    def get_train_dataset(self) -> List[Dict]:
        """
        Returns the training dataset as a list of dicts.
        Each dict contains 'prompt', 'response', 'class_label', 'reward'.
        """
        return [entry.to_dict() for entry in self.train_data]

    def get_eval_dataset(self) -> List[Dict]:
        """
        Returns the evaluation dataset similarly formatted.
        """
        return [entry.to_dict() for entry in self.eval_data]

    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Convenience method to get both train and eval datasets.
        """
        return self.get_train_dataset(), self.get_eval_dataset()
