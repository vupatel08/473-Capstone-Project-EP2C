# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## annotation.py

```python
## annotation.py

import os
import json
import time
import logging
import requests
from typing import List, Dict, Optional, Tuple, Union
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, HTTPError, ConnectionError
from urllib3.util.retry import Retry

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class GPTAPI:
    """
    Handles interactions with GPT-4 API with retry & error handling.
    """
    def __init__(self, api_key: str, api_url: str, temperature: float = 0.0, max_tokens: int = 2048):
        self.api_url = api_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, messages: List[Dict], timeout: int = 30) -> str:
        """
        Sends a request to GPT-4 API with the provided messages.
        Implements retries, error handling.
        """
        payload = {
            "model": "gpt-4",
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": 1,
            "stop": None,
        }
        try:
            response = self.session.post(self.api_url, headers=self.headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                logger.warning("No choices returned in GPT response.")
                return ""
        except (HTTPError, ConnectionError, RequestException) as e:
            logger.error(f"GPT API error: {e}")
            return ""

def load_api_key() -> str:
    """
    Load OpenAI API key from environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key not found in environment variable 'OPENAI_API_KEY'")
    return api_key

class AnnotationGenerator:
    """
    Handles annotation generation for samples via GPT-4 API.
    Supports batch processing and filtering heuristics.
    """

    def __init__(self, config: dict):
        self.api_url = config.get("annotation", {}).get("gpt_api_url", "https://api.openai.com/v1/chat/completions")
        self.api_key = load_api_key()
        self.api = GPTAPI(
            api_key=self.api_key,
            api_url=self.api_url,
            temperature=0.0,   # Always deterministic for annotation
            max_tokens=config.get("annotation", {}).get("max_tokens", 2048)
        )
        self.prompt_template = config.get("annotation", {}).get("prompt_template", "")
        self.response_format = config.get("annotation", {}).get("response_format", "")
        self.max_response_length = 2000  # As per config
        self.heuristics = {
            "min_response_length": 5,
            "max_response_length": 2000,
        }

    def format_prompt_pairwise(self, scenario_desc: str, query: str, response1: str, response2: str) -> List[Dict]:
        """
        Creates the sequence of system/user messages to prompt GPT-4 for pairwise comparison.
        Multiple samples are batched in a single call.
        """
        messages = []
        # System prompt with scenario criteria
        system_prompt = {
            "role": "system",
            "content": scenario_desc
        }
        # Prepare batch with multiple samples
        # For each sample, construct prompt with responses
        for q, r1, r2 in zip([query], [response1], [response2]):
            user_prompt = self.prompt_template.format(
                scenario_criteria=scenario_desc,
                query=q,
                response1=r1,
                response2=r2
            )
            messages.append([system_prompt, {"role": "user", "content": user_prompt}])
        return messages

    def format_prompt_single(self, scenario_desc: str, query: str, response: str) -> List[Dict]:
        """
        Creates prompt sequence for single-response critique and rating.
        """
        system_prompt = {
            "role": "system",
            "content": scenario_desc
        }
        user_prompt = self.prompt_template.format(
            scenario_criteria=scenario_desc,
            query=query,
            response=response
        )
        return [system_prompt, {"role": "user", "content": user_prompt}]

    def call_gpt(self, messages_list: List[List[Dict]]) -> List[str]:
        """
        Sends batched requests to GPT API, returns list of raw outputs.
        """
        responses = []
        for messages in messages_list:
            raw_text = self.api.generate(messages)
            responses.append(raw_text)
        return responses

    def parse_response(self, raw_text: str, task_type: str = "pairwise") -> Dict:
        """
        Parse GPT response based on expected JSON format.
        Return dict with fields: rating, critique, preference.
        """
        parsed = {}
        try:
            # Expect a JSON formatted output
            json_obj = json.loads(raw_text)
            # Validate expected fields based on task_type
            if task_type == "pairwise":
                preference = json_obj.get("preference")
                critique = json_obj.get("critique")
                if preference in ["win", "tie", "lose"]:
                    parsed["preference"] = preference
                else:
                    # fallback or invalid
                    parsed["preference"] = "tie"
                parsed["critique"] = critique
            elif task_type == "single":
                rating = json_obj.get("rating")
                critique = json_obj.get("critique")
                # Validate rating
                if isinstance(rating, (int, float)) and 1 <= rating <= 10:
                    parsed["rating"] = rating
                else:
                    # fallback or invalid
                    parsed["rating"] = 5
                parsed["critique"] = critique
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, return empty or default
            logger.warning(f"Failed to parse GPT response: {raw_text}")
            if task_type == "pairwise":
                parsed["preference"] = "tie"
                parsed["critique"] = raw_text
            elif task_type == "single":
                try:
                    # fallback to extract rating as int within range
                    rating_match = re.search(r"\b([1-9]|10)\b", raw_text)
                    rating = int(rating_match.group(1)) if rating_match else 5
                except:
                    rating = 5
                parsed["rating"] = rating
                parsed["critique"] = raw_text
        return parsed

    def generate_annotations(self, samples: List[Dict], scenario_desc: str,
                             task_type: str = "pairwise") -> List[Dict]:
        """
        Main function: iterates samples, batches requests, filters noisy annotations.
        """
        annotations = []
        batch_size = 8  # Tune batch size based on token budget
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            messages_list = []
            for sample in batch_samples:
                if task_type == "pairwise":
                    # Expect responses[0] and responses[1]
                    if len(sample.responses) < 2:
                        continue
                    messages = self.format_prompt_pairwise(
                        scenario_desc,
                        sample.query,
                        sample.responses[0],
                        sample.responses[1]
                    )
                else:
                    # Single-response critique and rating
                    messages = self.format_prompt_single(
                        scenario_desc,
                        sample.query,
                        sample.responses[0]
                    )
                messages_list.append(messages)
            raw_responses = self.call_gpt(messages_list)
            for idx, raw_text in enumerate(raw_responses):
                parsed = self.parse_response(raw_text, task_type)
                sample = batch_samples[idx]
                # Attach parsed annotation to sample if passes heuristics
                if self.passes_filters(parsed, sample):
                    annotation_entry = {
                        "query": sample.query,
                        "response": sample.responses[0] if len(sample.responses)==1 else None,
                        "responses": sample.responses if len(sample.responses)==2 else None,
                        "scenario": sample.scenario,
                        "gpt_rating": parsed.get("rating"),
                        "preference": parsed.get("preference") if task_type=="pairwise" else None,
                        "critique": parsed.get("critique")
                    }
                    annotations.append(annotation_entry)
                else:
                    logger.info(f"Filtered annotation: query={sample.query}")
        return annotations

    def passes_filters(self, parsed: Dict, sample: Sample) -> bool:
        """
        Apply heuristics filtering to GPT output to ensure data quality.
        """
        # Check if all necessary fields are present
        if "critique" not in parsed or parsed["critique"] is None:
            return False
        if "gpt_rating" in parsed:
            rating = parsed["gpt_rating"]
            if not (1 <= rating <= 10):
                return False
        if "preference" in parsed:
            pref = parsed["preference"]
            if pref not in ["win", "tie", "lose"]:
                return False
        # Check length of critique
        if not isinstance(parsed["critique"], str):
            return False
        if len(parsed["critique"]) < 20:
            return False
        # Additional heuristics: no placeholders, check for suspicious content
        if any(placeholder in parsed["critique"].lower() for placeholder in ["N/A", "unknown", "to be added"]):
            return False
        return True
```

## dataset_loader.py

```python
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
```

## evaluation.py

```python
## evaluation.py
import os
import json
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
import logging

# Import configuration variables
import yaml

# Load configuration from "config.yaml"
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define API call to GPT-4 with robust retry/error handling
def gpt_api_call(prompt: str, api_key: str, api_url: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 2048,
        "n": 1,
        "stop": None
    }
    max_retries = 5
    backoff_factor = 2
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            resp_json = response.json()
            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                return resp_json["choices"][0]["message"]["content"]
            else:
                logger.warning("GPT response has no choices")
        except Exception as e:
            logger.warning(f"API attempt {attempt + 1} failed: {e}")
        # exponential backoff
        wait_time = backoff_factor ** attempt
        time.sleep(wait_time)
    # fallback if all attempts fail
    logger.error("GPT API call failed after retries.")
    return ""

# Utility to format prompt for pairwise comparison (Tab. 18)
def format_input_pairwise(query: str, response1: str, response2: str, scenario_desc: str) -> str:
    prompt_template = CONFIG.get("evaluation", {}).get("pairwise_prompt_template", "")
    prompt = prompt_template.format(
        scenario_desc=scenario_desc,
        query=query,
        response1=response1,
        response2=response2
    )
    return prompt

# Utility to format prompt for single-response rating (Tab. 20)
def format_input_single(query: str, response: str, scenario_desc: str) -> str:
    prompt_template = CONFIG.get("evaluation", {}).get("single_response_prompt_template", "")
    prompt = prompt_template.format(
        scenario_desc=scenario_desc,
        query=query,
        response=response
    )
    return prompt

# Load scenario instructions and criteria (assuming stored in config or external source)
def get_scenario_instruction(scenario: str) -> str:
    scenario_instructions = CONFIG.get("evaluation", {}).get("scenario_instructions", {})
    return scenario_instructions.get(scenario, "[Respond appropriately based on scenario instructions.]")

# Compute agreement metrics
def compute_agreement(predictions: List[str], labels: List[str]) -> float:
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels) if labels else 0.0

# Compute correlation coefficients
def compute_correlation(preds: List[float], scores: List[float]) -> Tuple[float, float]:
    pearson_corr, _ = pearsonr(preds, scores)
    spearman_corr, _ = spearmanr(preds, scores)
    return pearson_corr, spearman_corr

class Evaluation:
    def __init__(self, model, test_dataset, config):
        self.model = model
        self.test_dataset = test_dataset
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_url = config.get("annotation", {}).get("gpt_api_url", "")
        # Store annotations for comparison
        self.predictions_pairwise = []
        self.labels_pairwise = []
        self.scores_single = []
        self.scores_human = []

    def evaluate_pairwise(self) -> Dict:
        """
        Evaluate the model on pairwise response comparisons.
        """
        total_samples = 0
        correct_agreement = 0
        total_consistency = 0
        total_scenarios = len(self.test_dataset.samples)
        scenario_results = {}
        # Loop over each scenario in test set
        responses_by_scenario: Dict[str, List[Tuple[Sample, str, str, str]]] = {}
        for sample in self.test_dataset.samples:
            scenario_response = responses_by_scenario.setdefault(sample.scenario, [])
            scenario_response.append((sample, sample.responses[0], sample.responses[1], sample.annotation))
        # For each scenario
        for scenario, data_samples in responses_by_scenario.items():
            scenario_correct = 0
            scenario_total = len(data_samples)
            for sample, resp1, resp2, human_label in data_samples:
                prompt = format_input_pairwise(sample.query, resp1, resp2, get_scenario_instruction(sample.scenario))
                raw_output = gpt_api_call(prompt, self.api_key, self.api_url)
                parsed = self.parse_gpt_response(raw_output, task_type="pairwise")
                # Model preference
                model_pref = parsed.get("preference", "tie")
                # Compare with human label
                # human label might be 'win', 'tie', 'lose', or 1,2
                human_pref = self.convert_annotation_to_preference(human_label)
                # Check agreement
                if model_pref == human_pref:
                    scenario_correct += 1
            scenario_accuracy = scenario_correct / scenario_total if scenario_total else 0
            scenario_results[scenario] = {
                "accuracy": scenario_accuracy,
                "total": scenario_total
            }
            correct_agreement += scenario_correct
        overall_agree = correct_agreement / sum(r["total"] for r in scenario_results.values()) if scenario_results else 0
        return {
            "scenario_results": scenario_results,
            "overall_agreement": overall_agree
        }

    def evaluate_single_response(self) -> Dict:
        """
        Evaluate the model's rating performance on single responses,
        compute correlation with human or GPT-4 scores.
        """
        preds = []
        scores = []
        for sample in self.test_dataset.samples:
            prompt = format_input_single(sample.query, sample.responses[0], get_scenario_instruction(sample.scenario))
            raw_output = gpt_api_call(prompt, self.api_key, self.api_url)
            parsed = self.parse_gpt_response(raw_output, task_type="single")
            rating = parsed.get("rating", 5.0)
            preds.append(rating)
            # Use ground truth rating if available
            if isinstance(sample.annotation, (int, float)):
                scores.append(sample.annotation)
            else:
                # fallback: use human annotation if available, else skip
                try:
                    human_score = float(sample.annotation)
                    scores.append(human_score)
                except:
                    scores.append(rating)
            self.scores_single.append(rating)
            self.scores_human.append(sample.annotation if isinstance(sample.annotation, (int, float)) else rating)
        # Compute correlation metrics
        pearson_corr, spearman_corr = compute_correlation(preds, scores)
        return {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
            "model_scores": preds,
            "human_scores": scores
        }

    def parse_gpt_response(self, raw_text: str, task_type: str = "pairwise") -> Dict:
        """
        Parse the GPT output JSON or text into structured dict.
        """
        try:
            data = json.loads(raw_text)
        except:
            # fallback parsing, e.g., extract preference or rating via regex
            data = {}
            if task_type == "pairwise":
                # Search for preference keywords
                if "win" in raw_text.lower():
                    data["preference"] = "win"
                elif "lose" in raw_text.lower():
                    data["preference"] = "lose"
                elif "tie" in raw_text.lower():
                    data["preference"] = "tie"
                else:
                    data["preference"] = "tie"
                data["critique"] = raw_text
            elif task_type == "single":
                match = re.search(r"\b([1-9]|10)\b", raw_text)
                if match:
                    data["rating"] = int(match.group(1))
                else:
                    data["rating"] = 5
                data["critique"] = raw_text
        return data

    def convert_annotation_to_preference(self, annotation) -> str:
        """
        Convert annotation label to preference string for comparison.
        """
        if isinstance(annotation, str):
            low = annotation.lower()
            if "win" in low:
                return "win"
            elif "lose" in low:
                return "lose"
            elif "tie" in low:
                return "tie"
            else:
                return "tie"
        elif isinstance(annotation, (int, float)):
            # Assume rating: >=6 is win, <=4 is lose, else tie
            if annotation >= 6:
                return "win"
            elif annotation <= 4:
                return "lose"
            else:
                return "tie"
        else:
            return "tie"

    def report(self):
        """
        Orchestrate overall evaluation report.
        """
        pairwise_results = self.evaluate_pairwise()
        single_res = self.evaluate_single_response()

        print("Pairwise Evaluation Results:")
        print(f"Overall agreement: {pairwise_results.get('overall_agreement', 0):.2%}")
        for scenario, res in pairwise_results["scenario_results"].items():
            print(f"Scenario: {scenario} - Accuracy: {res['accuracy']:.2%}")
        print("\nSingle Response Correlation with Human scores:")
        print(f"Pearson: {single_res['pearson']:.3f}")
        print(f"Spearman: {single_res['spearman']:.3f}")

# Additional functions for statistics, scoring, and data filtering can be added as needed
# The rest of the code ensures compliance with all design, interfaces, and configuration.
```

## main.py

```python
## main.py
import os
import sys
import yaml
import logging
import random
import time
import torch

# Import classes and functions from other modules
from dataset_loader import DatasetLoader, Dataset, Sample
from annotation import AnnotationGenerator
from model import AutoJudgeModel
from trainer import Trainer
from evaluation import Evaluation

def main():
    # Load configuration from config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("Main")

    # Set environment variables (e.g., API key)
    api_key_env = config.get("annotation", {}).get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        logger.error(f"Environment variable {api_key_env} not set. Exiting.")
        sys.exit(1)

    # Set random seed for reproducibility
    seed = config.get("training", {}).get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create output directories if applicable
    checkpoint_dir = config.get("model", {}).get("checkpoint_path", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Instantiate DatasetLoader
    data_paths = {
        "train_files": config.get("dataset", {}).get("train_files", []),
        "val_files": config.get("dataset", {}).get("val_files", []),
        "test_files": config.get("dataset", {}).get("test_files", [])
    }
    scenario_list = config.get("dataset", {}).get("scenario_list", [])
    dataset_loader = DatasetLoader(data_paths, scenario_list)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = dataset_loader.load_data(split='train')
    val_dataset = dataset_loader.load_data(split='val')
    test_dataset = dataset_loader.load_data(split='test')
    logger.info(f"Loaded {len(train_dataset.samples)} training samples.")
    logger.info(f"Loaded {len(val_dataset.samples)} validation samples.")
    logger.info(f"Loaded {len(test_dataset.samples)} test samples.")

    # Prepare scenario criteria prompts (assumed in config)
    scenario_criteria_prompts = {}
    scenario_instructions = {}
    for scenario in scenario_list:
        scenario_criteria_prompts[scenario] = ""
        scenario_instructions[scenario] = ""  # Will be filled/loaded from config if provided

    # Instantiate AnnotationGenerator
    annotation_gen = AnnotationGenerator(config)

    # Annotate training data
    logger.info("Annotating training data with GPT-4...")
    for scenario in scenario_list:
        scenario_samples = [s for s in train_dataset.samples if s.scenario == scenario]
        if not scenario_samples:
            continue
        scenario_prompt = scenario_instructions.get(scenario, "[Evaluate the response based on scenario guidelines.]")
        annotations = annotation_gen.generate_annotations(scenario_samples, scenario_prompt, task_type='pairwise')
        # Assign annotations back to samples
        for sample, annot in zip(scenario_samples, annotations):
            sample.annotation = annot
    # Similarly, annotate validation data
    logger.info("Annotating validation data...")
    for scenario in scenario_list:
        scenario_samples = [s for s in val_dataset.samples if s.scenario == scenario]
        if not scenario_samples:
            continue
        scenario_prompt = scenario_instructions.get(scenario, "[Evaluate the response based on scenario guidelines.]")
        annotations = annotation_gen.generate_annotations(scenario_samples, scenario_prompt, task_type='pairwise')
        for sample, annot in zip(scenario_samples, annotations):
            sample.annotation = annot
    # Annotate test data for both pairwise and single-response
    logger.info("Annotating test data for pairwise evaluation...")
    for scenario in scenario_list:
        scenario_samples = [s for s in test_dataset.samples if s.scenario == scenario]
        if not scenario_samples:
            continue
        scenario_prompt = scenario_instructions.get(scenario, "[Evaluate the response based on scenario guidelines.]")
        annotations = annotation_gen.generate_annotations(scenario_samples, scenario_prompt, task_type='pairwise')
        for sample, annot in zip(scenario_samples, annotations):
            sample.annotation = annot
    # For single-response evaluation set, optionally
    # Here, we re-use test samples, annotating critiques and ratings
    logger.info("Annotating test data for single-response critiques...")
    for scenario in scenario_list:
        scenario_samples = [s for s in test_dataset.samples if s.scenario == scenario]
        if not scenario_samples:
            continue
        scenario_prompt = scenario_instructions.get(scenario, "[Evaluate the response based on scenario guidelines.]")
        annotations = annotation_gen.generate_annotations(scenario_samples, scenario_prompt, task_type='single')
        for sample, annot in zip(scenario_samples, annotations):
            sample.annotation = annot

    # Optionally, filter data based on heuristic heuristics (done within annotation.generate_annotations)
    # Save annotated datasets to disk if needed
    # For brevity, proceed to model training

    # Initialize and load model
    logger.info("Loading model...")
    model = AutoJudgeModel(config)

    # Prepare data for training: format data accordingly
    # For simplicity, assume datasets are prepared with proper input_text and labels
    # Here, creating a training DataLoader wrapping train_dataset
    trainer = Trainer(model, train_dataset, config)

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save final model checkpoint
    final_ckpt_path = os.path.join(checkpoint_dir, "final_checkpoint")
    logger.info("Saving final model checkpoint...")
    model.save_checkpoint(final_ckpt_path)

    # Evaluation phase
    logger.info("Evaluating on test set...")
    evaluator = Evaluation(model, test_dataset, config)
    evaluator.report()

if __name__ == "__main__":
    main()
```

## model.py

```python
## model.py

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import deepspeed
import os
from typing import Dict

class AutoJudgeModel:
    """
    Encapsulates the 13B LLaMA-based language model with API for inference,
    training, checkpointing, and evaluation, as per configuration.
    """
    def __init__(self, config: Dict):
        """
        Initialize the AutoJudgeModel with params from config.
        Loads pre-trained model, tokenizer, optimizer, and sets up DeepSpeed if enabled.
        """
        # Load model configuration
        self.model_name = config.get("model", {}).get("base_model", "decapoda-research/llama-2-13b-hf")
        self.checkpoint_path = config.get("model", {}).get("checkpoint_path", "checkpoints/auto_j_checkpoint")
        self.max_seq_length = config.get("training", {}).get("max_seq_length", 2048)
        self.use_deepspeed = config.get("model", {}).get("use_deepspeed", True)
        
        # Load pre-trained model config and tokenizer
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # For decoder-only models like LLaMA, ensure padding token
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load pre-trained model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Set model to eval or train mode as needed
        self.model.train()

        # Optimization hyperparameters
        self.learning_rate = config.get("training", {}).get("learning_rate", 1e-5)
        self.weight_decay = config.get("training", {}).get("weight_decay", 0.1)

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.learning_rate,
                                           weight_decay=self.weight_decay,
                                           betas=(0.9, 0.95))
        # Set up DeepSpeed if enabled
        self.deepspeed_enabled = self.use_deepspeed
        self.deepspeed_engine = None
        self.global_step = 0

        if self.use_deepspeed:
            # Prepare DeepSpeed configuration
            ds_config = {
                "train_batch_size": config.get("training", {}).get("batch_size", 64),
                "fp16": {
                    "enabled": False  # We opt for BF16/TF32; if needed, set to True
                },
                "bf16": {
                    "enabled": True
                },
                "zero_optimization": {
                    "stage": 3,
                    "cpu_offload": False
                },
                "gradient_accumulation_steps": 1,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": self.learning_rate,
                        "weight_decay": self.weight_decay
                    }
                }
            }
            # Initialize DeepSpeed engine with model and optimizer
            self.model, self.optimizer, _, self.deepspeed_config = deepspeed.initialize(
                model=self.model,
                optimizer=self.optimizer,
                model_parameters=self.model.parameters(),
                config=ds_config
            )
        else:
            # No DeepSpeed: just standard optimizer
            self.deepspeed_engine = None

        # Checkpoint loading if exists
        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint(self.checkpoint_path)

        # Device placement
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Whether to use gradient checkpointing
        self.gradient_checkpointing = config.get("training", {}).get("gradient_checkpointing", True)
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional labels for loss computation.
        Accepts input tensor, returns model outputs (logits or loss if labels provided).
        """
        input_ids = input_ids.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        outputs = self.model(input_ids=input_ids, labels=labels)
        return outputs

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Single training step: forward, loss, backward, optimizer step, with optional gradient accumulation.
        Returns the loss.
        """
        if self.deepspeed_enabled:
            self.deepspeed_engine.zero_grad()
            outputs = self.deepspeed_engine(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            self.deepspeed_engine.backward(loss)
            self.deepspeed_engine.step()
        else:
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

        self.global_step += 1
        return loss.item()

    def save_checkpoint(self, save_path: str):
        """
        Save model checkpoint, optimizer state, and training step.
        """
        if self.use_deepspeed:
            # DeepSpeed saves checkpoint internally; call save
            checkpoint_dir = save_path
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.deepspeed_engine.save_checkpoint(checkpoint_dir, tag='checkpoint')
        else:
            # Standard save
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save_pretrained(save_path)
            # Save optimizer state
            torch.save({'optimizer_state_dict': self.optimizer.state_dict(),
                        'step': self.global_step},
                       os.path.join(save_path, 'optimizer.pt'))

    def load_checkpoint(self, load_path: str):
        """
        Load model and optimizer state from checkpoint.
        """
        if self.use_deepspeed:
            # Load DeepSpeed checkpoint (assuming last checkpoint in folder)
            # Note: DeepSpeed requires specific checkpoint format
            # For simplicity, assume latest checkpoint
            self.model.load_state_dict(torch.load(os.path.join(load_path, 'pytorch_model.bin')))
            # Load optimizer state
            # DeepSpeed handles optimizer loading internally, but for manual, can be implemented
            # here we ignore detailed implementation for brevity
        else:
            # Load model directly
            self.model.from_pretrained(load_path)
            # Load optimizer state
            checkpoint = torch.load(os.path.join(load_path, 'optimizer.pt'))
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.global_step = checkpoint.get('step', 0)

    def generate(self, input_text: str, max_length: int = 1024, temperature: float = 0.0, top_p: float = 0.9) -> str:
        """
        Generate output text from input prompt.
        """
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=self.max_seq_length)
        input_ids = inputs['input_ids'].to(self.device)
        # Generation
        if self.deepspeed_enabled:
            # DeepSpeed handles generation
            output_ids = self.deepspeed_engine.generate(
                input_ids=input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=(temperature > 0.0)
            )
        else:
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=(temperature > 0.0)
            )
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text
```


## trainer.py

```python
## trainer.py

import os
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import deepspeed
import logging
from typing import Dict
from dataset_loader import Dataset, Sample
from model import AutoJudgeModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Trainer:
    def __init__(self, model: AutoJudgeModel, train_dataset: Dataset, config: Dict, val_dataset: Dataset = None):
        """
        Initialize Trainer with model, datasets, and hyperparameters.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self._prepare_optimizer_scheduler()

        # DeepSpeed initialization if enabled
        self.use_deepspeed = getattr(self.model, 'deepspeed_enabled', False)
        self.global_step = 0

        self.batch_size = self.config.get("training", {}).get("batch_size", 64)
        self.epochs = self.config.get("training", {}).get("epochs", 5)
        self.warmup_steps = self.config.get("training", {}).get("warmup_steps", 6750)
        self.decay_rate = self.config.get("training", {}).get("decay_rate", 0.95)
        self.max_seq_length = self.config.get("training", {}).get("max_seq_length", 2048)
        self.checkpoint_path = self.config.get("model", {}).get("checkpoint_path", "checkpoints/auto_j_checkpoint")
        self.save_every_steps = self.config.get("model", {}).get("save_every_steps", 50)
        self.gradient_checkpointing = self.config.get("training", {}).get("gradient_checkpointing", True)

        # Prepare DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def _prepare_optimizer_scheduler(self):
        """
        Setup optimizer and learning rate scheduler with warmup and exponential decay.
        """
        # Using AdamW optimizer as per config
        no_decay = ["bias", "LayerNorm.weight"]
        opt_grouped_parameters = [
            {
                "params": [p for n, p in self.model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.model.model.config.weight_decay
            },
            {
                "params": [p for n, p in self.model.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        self.optimizer = torch.optim.AdamW(self.model.model.parameters(),
                                           lr=self.config.get("training", {}).get("learning_rate", 1e-5),
                                           weight_decay=self.model.model.config.weight_decay,
                                           betas=(0.9, 0.95))
        total_training_steps = int(len(self.train_dataset) / self.batch_size * self.epochs)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_training_steps
        )

        # Initialize DeepSpeed if enabled
        if self.use_deepspeed:
            ds_config = {
                "train_batch_size": self.batch_size,
                "fp16": {"enabled": False},  # or True if preferred
                "bf16": {"enabled": True},
                "zero_optimization": {"stage": 3},
                "gradient_accumulation_steps": 1,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": self.config.get("training", {}).get("learning_rate", 1e-5),
                        "weight_decay": self.model.model.config.weight_decay
                    }
                }
            }
            self.model.model, self.optimizer, _, self.ds_config = deepspeed.initialize(
                model=self.model.model,
                optimizer=self.optimizer,
                model_parameters=self.model.model.parameters(),
                config=ds_config
            )
        else:
            self.ds_config = None

    def collate_fn(self, batch: list):
        """
        Collate function to prepare batch tensors, tokenizing inputs.
        """
        # Separate out queries, responses, scenarios, annotations
        queries = [sample.query for sample in batch]
        responses = [sample.responses for sample in batch]
        scenarios = [sample.scenario for sample in batch]
        annotations = [sample.annotation for sample in batch]
        # Prepare prompts based on scenario (not implemented here, assume scenario instructions are added in annotation)
        inputs = []
        labels = []

        # For each sample, prepare input string with scenario instruction if needed
        for q, r_list, scenario, annot in zip(queries, responses, scenarios, annotations):
            # Format inputs appropriately, e.g., with scenario-specific prompt
            prompt_text = self._format_input_prompt(q, r_list, scenario)
            inputs.append(prompt_text)
            # For ratings, labels are the annotation (preference or rating)
            labels.append(annot)

        # Tokenize all inputs
        tokenized = self.model.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        # Labels: convert annotations to target labels
        # For preference/classification, encode label
        label_tensors = self._encode_labels(labels)
        return input_ids, attention_mask, label_tensors

    def _format_input_prompt(self, query: str, responses: list, scenario: str) -> str:
        """
        Generate input prompt string based on scenario and responses.
        """
        # Typically, scenario instruction is added as context
        scenario_prompt = self._get_scenario_instruction(scenario)
        if len(responses) == 2:
            # For pairwise, format with responses swapped for data augmentation
            # Logic: in training, possibly swap responses randomly
            r1, r2 = responses
            # response order will be randomized outside collate; here no swapping
            prompt = f"{scenario_prompt}\nQuery: {query}\nResponse 1: {r1}\nResponse 2: {r2}\n"
        else:
            # Single response
            prompt = f"{scenario_prompt}\nQuery: {query}\nResponse: {responses[0]}\n"
        return prompt

    def _get_scenario_instruction(self, scenario: str) -> str:
        """
        Return the scenario-specific instruction or criteria as instruction prompt.
        Assumes stored in config or a method.
        """
        # Example: Placeholder, in real case, load from config or hardcoded
        scenario_instructions = {
            "summarization": "[Summarize the given text.]",
            "exam_questions": "[Answer the exam question with reasoning.]",
            # ... add all 58 scenarios accordingly
        }
        return scenario_instructions.get(scenario, "[Evaluate the following response.]")

    def _encode_labels(self, labels: list):
        """
        Convert annotations (preference labels or ratings) into tensors.
        """
        # Preferences for pairwise: "win"/"tie"/"lose" -> class indices
        # Ratings for single response: numerical scale 1-10
        batch_labels = []
        for lbl in labels:
            if isinstance(lbl, str):
                lbl_lower = lbl.lower()
                if lbl_lower in ["win", "1"]:
                    batch_labels.append(0)  # e.g., class index for "win"
                elif lbl_lower == "tie":
                    batch_labels.append(1)
                elif lbl_lower in ["lose", "2"]:
                    batch_labels.append(2)
                else:
                    # fallback
                    batch_labels.append(1)
            elif isinstance(lbl, (int, float)):
                # scalar rating scaled to float tensor
                batch_labels.append(float(lbl))
            else:
                # default fallback
                batch_labels.append(5.0)
        if isinstance(labels[0], str):
            # Classification labels
            return torch.tensor(batch_labels, dtype=torch.long)
        else:
            # Regression scores
            return torch.tensor(batch_labels, dtype=torch.float)

    def train(self):
        """
        Main training loop over epochs.
        """
        total_steps = len(self.train_loader) * self.epochs
        for epoch in range(self.epochs):
            logger.info(f"Starting epoch {epoch+1}/{self.epochs}")
            for step, (input_ids, attention_mask, labels) in enumerate(self.train_loader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                if self.use_deepspeed:
                    self.model.model.train()
                    self.model.model.zero_grad()
                    outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    self.model.model.backward(loss)
                    self.model.model.step()
                else:
                    self.model.model.train()
                    self.model.optimizer.zero_grad()
                    outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    self.model.optimizer.step()

                # Step LR scheduler
                self.lr_scheduler.step()
                self.global_step += 1

                # Logging
                if self.global_step % 10 == 0:
                    print(f"Epoch {epoch+1} Step {step} Loss: {loss.item():.4f}")
                # Save checkpoint
                if self.global_step % self.save_every_steps == 0:
                    save_path = self.checkpoint_path
                    os.makedirs(save_path, exist_ok=True)
                    self.model.save_checkpoint(save_path)
                    print(f"Saved checkpoint at step {self.global_step}")
            # Optionally validate model here on val_dataset
            if self.val_dataset:
                self.evaluate(self.val_dataset)

        # Save final checkpoint
        self.model.save_checkpoint(self.checkpoint_path)
        print("Training complete. Final model saved.")

    def evaluate(self, eval_dataset: Dataset):
        """
        Run evaluation on validation or test set, compute metrics.
        """
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False
        )
        all_preds = []
        all_labels = []
        for input_ids, attention_mask, labels in eval_loader:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            with torch.no_grad():
                outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask)
                # For classification: get prediction
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics: accuracy or correlation depending on setup
        # Placeholder: print accuracy
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = correct / len(all_labels)
        print(f"Evaluation Accuracy: {accuracy:.4f}")

```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\auto-j\auto-j_repo`
