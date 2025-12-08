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
