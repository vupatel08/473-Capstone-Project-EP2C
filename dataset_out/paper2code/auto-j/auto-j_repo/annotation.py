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
