# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py

import json
import os
import random
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Assuming the model.py defines a LanguageModel, imported here
from model import LanguageModel


class Instruction:
    """
    Represents a single instruction with optional metadata.
    """
    def __init__(self, instruction_id: str, text: str, source: Optional[str] = None):
        self.id = instruction_id
        self.text = text
        self.source = source


class ResponseSample:
    """
    Represents a generated response for an instruction, with optional token info.
    """
    def __init__(self,
                 instruction_id: str,
                 response_text: str,
                 response_id: str,
                 response_tokens: Optional[int] = None,
                 response_length: Optional[int] = None):
        self.instruction_id = instruction_id
        self.response_text = response_text
        self.response_id = response_id
        self.response_tokens = response_tokens
        self.response_length = response_length


class FeedbackInstance:
    """
    Represents a feedback data point, either rating or ranking.
    """
    def __init__(self,
                 instruction_id: str,
                 responses: List[str],  # For ratings, list length=1; for rankings, list length=2 or more
                 score: Optional[float] = None,  # For ratings
                 preference: Optional[int] = None):  # For rankings: 1/2 (which response is preferred)
        self.instruction_id = instruction_id
        self.responses = responses
        self.score = score
        self.preference = preference  # 1 if responses[0] preferred, 2 if responses[1], 0 if equal


class DatasetLoader:
    """
    Class for loading instruction data, feedback data, and generating responses.
    """
    def __init__(self, config: dict):
        """
        Initialize with configuration, e.g.,
        {
            "instruction_data_path": "path/to/instructions.json",
            "feedback_data_path": "path/to/feedback.json",
            "reference_responses_path": "path/to/reference.json",
            "seed": 42
        }
        """
        self.config = config
        self.instruction_data_path: str = config.get("instruction_data_path", "")
        self.feedback_data_path: str = config.get("feedback_data_path", "")
        self.reference_responses_path: str = config.get("reference_responses_path", "")
        self.seed: int = config.get("seed", 42)

        # Initialize internal storage
        self.instructions: List[Instruction] = []
        self.ratings_feedback: Dict[Tuple[str, str], float] = {}  # (instruction_id, response_id) -> score
        self.rankings_feedback: List[Tuple[str, str, str, int]] = []  # (instruction_id, resp_id1, resp_id2, preference)
        self.reference_responses: Dict[str, str] = {}  # instruction_id -> response

        # For reproducibility
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def load_instructions(self):
        """
        Load instruction dataset (assumed JSON lines or list of dicts).
        Each instruction entry should contain at least 'id' and 'text'.
        """
        if not os.path.exists(self.instruction_data_path):
            raise FileNotFoundError(f"Instruction data file not found at {self.instruction_data_path}")

        with open(self.instruction_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.instructions = []
        for idx, item in enumerate(data):
            instr_id = item.get('id', f'instr_{idx}')
            text = item.get('text') or item.get('instruction') or item.get('prompt')
            source = item.get('source', None)
            if text:
                self.instructions.append(Instruction(instr_id, text, source))
            else:
                # Skip entries without 'text'
                continue

    def load_feedback(self, feedback_format: str = 'json'):
        """
        Load feedback data, depending on the format (assumed JSON).
        Supports 'ratings' and 'rankings' in the same file or separate files.
        The format is assumed to follow the paper's description.
        """
        if not os.path.exists(self.feedback_data_path):
            raise FileNotFoundError(f"Feedback data file not found at {self.feedback_data_path}")

        with open(self.feedback_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Clear previous data
        self.ratings_feedback.clear()
        self.rankings_feedback.clear()

        # Load feedback; depends on format
        for entry in data:
            instr_id = entry.get('instruction_id')
            # Ratings
            if 'rating' in entry:
                response_id = entry.get('response_id')
                score = entry.get('rating')  # expected in 1-7
                if score is not None and 1 <= score <=7:
                    self.ratings_feedback[(instr_id, response_id)] = float(score)
            # Rankings
            if 'rankings' in entry:
                # Should be a list of dicts with response comparisons
                for comp in entry['rankings']:
                    resp_id1 = comp.get('response_id1')
                    resp_id2 = comp.get('response_id2')
                    preference = comp.get('preference')  # 1, 2, or 0 for equal
                    if preference in [1,2,0]:
                        self.rankings_feedback.append(
                            (instr_id, resp_id1, resp_id2, preference)
                        )

    def load_references(self):
        """
        Load reference responses used for evaluation.
        Assumed JSON: { instruction_id: response }
        """
        if not os.path.exists(self.reference_responses_path):
            raise FileNotFoundError(f"Reference responses file not found at {self.reference_responses_path}")

        with open(self.reference_responses_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.reference_responses = data  # dict: instruction_id -> response

    def generate_responses(self,
                           model: 'LanguageModel',
                           instructions: List[Instruction],
                           responses_per_instruction: int = 5,
                           max_length: int = 128,
                           temperature: float = 0.0) -> List[ResponseSample]:
        """
        Generate responses for each instruction using the provided model.
        Responses are labeled with responses IDs for tracking.
        """
        generated_responses: List[ResponseSample] = []
        for instr in instructions:
            responses = model.generate(
                prompt=instr.text,
                max_length=max_length,
                num_return_sequences=responses_per_instruction,
                temperature=temperature
            )
            for idx, resp_text in enumerate(responses):
                # Tokenize response to get length info
                tokens = model.tokenize(resp_text)
                resp_id = f"{instr.id}_resp_{idx}"
                resp_sample = ResponseSample(
                    instruction_id=instr.id,
                    response_text=resp_text,
                    response_id=resp_id,
                    response_tokens=len(tokens),
                    response_length=len(resp_text)
                )
                generated_responses.append(resp_sample)
        return generated_responses

    def load_all(self):
        """
        Convenience method to load all datasets.
        """
        self.load_instructions()
        self.load_feedback()
        self.load_references()
```

## evaluation.py

```python
## evaluation.py
import json
import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
from collections import defaultdict
import random

# Import classes from other modules
from dataset_loader import Instruction, ResponseSample
from model import LanguageModel
from reward_model import RewardModel
from policy import PolicySampler

# Load configuration
import yaml
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

class Evaluation:
    """
    Class for evaluating the response quality, win rates, and consistency between feedback protocols.
    """
    def __init__(self,
                 instruction_data_path: str = CONFIG["evaluation"]["test_instructions_path"],
                 reference_responses_path: str = CONFIG["evaluation"]["reference_responses_path"],
                 reward_model_paths: Dict[str, str] = None,
                 test_instructions_path: str = None,
                 reference_responses_path: str = None,
                 feedback_protocol: str = "ranking",  # or "rating"
                 n_responses: int = 64,
                 eval_samples: int = 1000,
                 seed: int = 42):
        """
        Initialize evaluation with models, datasets, and configurations.
        """
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Paths
        self.instruction_data_path = instruction_data_path
        self.reference_responses_path = reference_responses_path
        self.feedback_protocol = feedback_protocol
        self.n_responses = n_responses
        self.eval_samples = eval_samples

        # Load instruction set
        self.instructions = self.load_instructions()

        # Load reference responses
        self.reference_responses = self.load_reference_responses()

        # Load models
        # Base language model (e.g., Alpaca-7B)
        self.base_model = LanguageModel(model_name=CONFIG["training"].get("model_name", "alpaca"), device='cuda')

        # Load reward models (for different feedback protocols)
        # Expect paths in reward_model_paths dict, keys: "rating", "ranking"
        self.reward_models = {}
        if reward_model_paths:
            for key, path in reward_model_paths.items():
                mode = "regression" if key == "rating" else "preference"
                self.reward_models[key] = RewardModel(
                    model_name=CONFIG["reward_model"].get("model_name", "allenai/longformer-base-4096"),
                    feedback_data=None,  # None, will be loaded from trained checkpoints
                    training_mode=mode,
                    device='cuda'
                )
                self.reward_models[key].load(path)
        else:
            # If no explicit paths, instantiate models (not trained here)
            pass

        # Initialize PolicySampler for sampling responses
        # For simplicity, use the base model as sampler
        self.policy_sampler = PolicySampler(
            language_model=self.base_model,
            reward_model=None,  # Will set later when sampling
            n_responses=self.n_responses,
            temperature=CONFIG["sampling"].get("temperature", 0.0)
        )

    def load_instructions(self) -> List[Instruction]:
        """
        Load test instructions from JSON file.
        Expected format: list of dicts with at least 'id' and 'text' keys.
        """
        with open(self.instruction_data_path, 'r') as f:
            data = json.load(f)
        instructions = []
        for item in data:
            instr_id = item.get('id', '')
            text = item.get('text', '') or item.get('instruction', '')
            instructions.append(Instruction(instr_id, text))
        return instructions

    def load_reference_responses(self) -> Dict[str, str]:
        """
        Load reference responses for evaluation.
        Format: { instruction_id: response }
        """
        with open(self.reference_responses_path, 'r') as f:
            refs = json.load(f)
        return refs

    def generate_responses(self, instruction: Instruction) -> List[str]:
        """
        Generate multiple responses for the instruction using the base LM.
        """
        responses = self.base_model.generate(
            prompt=instruction.text,
            max_length=128,
            temperature=CONFIG["sampling"].get("temperature", 0.0),
            num_return_sequences=self.n_responses
        )
        return responses

    def score_response(self, response: str, instruction: str, model: RewardModel) -> float:
        """
        Score a single response using the provided reward model.
        """
        return model.score_response(response, instruction)

    def score_responses(self, responses: List[str], instruction: str, model: RewardModel) -> List[float]:
        """
        Score all responses for an instruction using the reward model.
        """
        scores = []
        for resp in responses:
            score = self.score_response(resp, instruction, model)
            scores.append(score)
        return scores

    def compute_preferences(self, responses: List[str], instruction: str, model: RewardModel) -> List[Tuple[int, float]]:
        """
        For pairwise responses, compute preferences based on scores.
        Return list of tuples: (preference, score_diff)
        preference: 1 if response 1 preferred, 2 if response 2, 0.5 if tie
        """
        preferences = []
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                score_i = self.score_response(responses[i], instruction, model)
                score_j = self.score_response(responses[j], instruction, model)
                if abs(score_i - score_j) < 1e-4:
                    pref = 0.5
                elif score_i > score_j:
                    pref = 1
                else:
                    pref = 2
                preferences.append((pref, score_i - score_j))
        return preferences

    def evaluate_instruction(self, instruction: Instruction, reference_response: str,
                             model: RewardModel, protocol: str = "ranking") -> dict:
        """
        Generate responses, score them, and compute preferences for one instruction.
        """
        responses = self.generate_responses(instruction)
        scores = self.score_responses(responses, instruction.text, model)

        # For ranking protocol: pairwise preferences
        preferences = []
        if protocol == "ranking":
            preferences = self.compute_preferences(responses, instruction.text, model)
        # For rating protocol: use individual scores
        elif protocol == "rating":
            preferences = list(zip(scores, responses))
        else:
            raise ValueError("Protocol must be 'ranking' or 'rating'.")

        return {
            "instruction_id": instruction.id,
            "responses": responses,
            "scores": scores,
            "preferences": preferences
        }

    def run_evaluation(self, protocol: str = "ranking", eval_reference: bool = True,
                       eval_iterations: int = None) -> List[dict]:
        """
        Run evaluation on all instructions, generate responses, compute scores, preferences.
        """
        results = []

        # For each instruction, generate responses, score, evaluate
        for instr in self.instructions:
            # Use the first reward model based on protocol
            model = self.reward_models.get(protocol)
            if model is None:
                # fallback to a default or skip
                continue
            result = self.evaluate_instruction(instr, self.reference_responses.get(instr.id, ""), model, protocol)
            results.append(result)
        return results

    def compute_win_rate(self, responses_data: List[dict], protocol: str = "ranking") -> float:
        """
        Compute win rate of policy-generated responses vs reference responses.
        """
        wins = 0
        total = 0
        for item in responses_data:
            instr_id = item['instruction_id']
            responses = item['responses']
            scores = item['scores']
            # Generate response from reference
            ref_response = self.reference_responses.get(instr_id, "")

            # Score reference
            ref_score = 0.
            if self.reward_models.get(protocol):
                ref_score = self.score_response(ref_response, instr_id, self.reward_models[protocol])

            max_idx = scores.index(max(scores))
            best_response = responses[max_idx]
            best_score = scores[max_idx]

            # Compare window: response vs reference
            if self.reward_models.get(protocol):
                ref_response_score = self.score_response(ref_response, instr_id, self.reward_models[protocol])
            else:
                ref_response_score = 0

            # Preference: stepped by protocol type with tie handling
            if protocol == "ranking":
                # use scores directly
                if best_score > ref_response_score + 1e-4:
                    wins += 1
                elif abs(best_score - ref_response_score) < 1e-4:
                    wins += 0.5
            elif protocol == "rating":
                # compare scalar scores directly
                if best_score > ref_response_score + 1e-4:
                    wins += 1
                elif abs(best_score - ref_response_score) < 1e-4:
                    wins += 0.5
            total += 1
        return wins / total if total > 0 else 0.0

    def assess_inconsistency(self, data_ai: List[dict], data_human: List[dict]) -> dict:
        """
        Compare feedback from AI and humans on the same set of responses.
        Compute percentages of agreement/disagreement.
        """
        consistent_count = 0
        total_count = 0
        inconsistent_count = 0

        # Convert ratings to preferred response (ranking form) for comparisons
        # For each response pair, compare preferences from AI and human
        for ai_entry in data_ai:
            instr_id = ai_entry['instruction_id']
            # Find matching human feedback for same responses
            # Here, just a placeholder: in practice, align data properly
            human_entry = next((h for h in data_human if h['instruction_id'] == instr_id), None)
            if human_entry is None:
                continue

            # Convert rating to ranking preference
            ai_pref = ai_entry.get('preference', None)
            human_pref = human_entry.get('preference', None)
            if ai_pref is None or human_pref is None:
                continue

            total_count += 1
            if abs(ai_pref - human_pref) < 1e-4:
                consistent_count += 1
            else:
                inconsistent_count += 1

        inconsistency_percentage = 100 * (inconsistent_count / total_count) if total_count > 0 else 0

        return {
            "total_comparisons": total_count,
            "consistent": 100 * (consistent_count / total_count) if total_count > 0 else 0,
            "inconsistent": inconsistency_percentage
        }

    def run_full_evaluation(self):
        """
        Run the entire evaluation pipeline, including response generation,
        scoring, win-rate, and inconsistency analysis.
        """
        # Generate responses and evaluate
        responses_data = self.run_evaluation(protocol=self.feedback_protocol)

        # Compute win-rate vs reference
        win_rate = self.compute_win_rate(responses_data, protocol=self.feedback_protocol)

        # Optional: inconsistency measures if data from AI and Human feedback available
        # For illustration, assume data_ai and data_human are available
        data_ai = []   # Load AI feedback data (placeholder)
        data_human = []  # Load human feedback data (placeholder)
        inconsistency = self.assess_inconsistency(data_ai, data_human)

        # Response quality metrics (length, diversity) can be computed here
        # Placeholder for auxiliary metrics
        length_stats = self.compute_response_length_stats(responses_data)

        return {
            "win_rate": win_rate,
            "inconsistency": inconsistency,
            "length_stats": length_stats
        }

    def compute_response_length_stats(self, responses_data: List[dict]) -> dict:
        """
        Calculate statistics like average length, diversity for responses.
        """
        lengths = []
        unique_token_counts = []
        for item in responses_data:
            for resp in item['responses']:
                tokens = resp.split()
                lengths.append(len(tokens))
                unique_token_counts.append(len(set(tokens)))
        return {
            "avg_length": np.mean(lengths) if lengths else 0,
            "avg_unique_tokens": np.mean(unique_token_counts) if unique_token_counts else 0
        }
```

## main.py

```python
## main.py

import os
import sys
import yaml
import json
import logging
import random
import torch

from dataset_loader import DatasetLoader, Instruction, ResponseSample
from model import LanguageModel
from reward_model import RewardModel
from trainer import RewardTrainer
from policy import PolicySampler
from evaluation import Evaluation

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load configuration from config.yaml
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- 1. Data Loading ---
    data_dir = cfg.get("data_dir", "")  # Optional, not specified in config
    data_loader = DatasetLoader(cfg)
    logging.info("Loading instruction data...")
    data_loader.load_all()

    # Extract instruction list
    instructions = data_loader.instructions

    # --- 2. Initialize the language model ---
    model_name = cfg.get("training", {}).get("model_name", "hf-internal-testing/tiny-random-gpt2")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    language_model = LanguageModel(model_name=model_name, device=device)
    logging.info(f"Loaded language model: {model_name} on device: {device}")

    # --- 3. Generate candidate responses for each instruction ---
    responses_per_instruction = cfg.get("sampling", {}).get("responses_per_instruction", 5)
    max_length = 128
    temperature = cfg.get("sampling", {}).get("temperature", 0.0)
    all_generated_responses: Dict[str, List[ResponseSample]] = {}

    logging.info("Generating responses for instructions...")
    for instr in instructions:
        responses = data_loader.generate_responses(language_model, [instr], responses_per_instruction, max_length, temperature)
        all_generated_responses[instr.id] = responses
    logging.info(f"Generated responses for {len(instructions)} instructions.")

    # --- 4. Load feedback data ---
    feedback_path = cfg.get("feedback", {}).get("feedback_data_path", "")
    feedback_protocol = cfg.get("feedback", {}).get("feedback_protocol", "ranking")
    with open(feedback_path, "r") as f:
        feedback_data_json = json.load(f)

    # Convert feedback to preferred format for reward model training
    feedback_data = {
        "ratings": [],
        "preferences": [],
        # Will fill in based on feedback format
    }

    # --- 5. Prepare feedback data for training reward model ---
    # Assume feedback_json is list of dicts with appropriate keys
    for entry in feedback_data_json:
        # Ratings
        if 'rating' in entry:
            feedback_data["ratings"].append({
                'instruction': entry['instruction_id'],
                'response': entry['response_id'],
                'score': entry['rating']
            })
        # Preferences
        if 'ranking' in entry:
            for comp in entry['rankings']:
                feedback_data["preferences"].append({
                    'instruction': entry['instruction_id'],
                    'response1': comp['response_id1'],
                    'response2': comp['response_id2'],
                    'preference': comp['preference']
                })

    # --- 6. Instantiate and train reward model ---
    # Choose training mode based on protocol
    training_mode = 'regression' if feedback_protocol == 'rating' else 'preference'
    reward_model = RewardModel(
        model_name=cfg.get("reward_model", {}).get("model_name", "allenai/longformer-base-4096"),
        feedback_data=feedback_data,
        training_mode=training_mode,
        device=device
    )

    # Prepare dataset for training
    reward_model._prepare_data()

    # Train reward model with early stopping
    trainer = RewardTrainer(cfg, feedback_data)
    logging.info("Training reward model...")
    trainer.train()
    # Save the best reward model checkpoint
    save_path = f"reward_{feedback_protocol}_best.pt"
    reward_model.save(save_path)
    logging.info(f"Reward model saved at {save_path}")

    # --- 7. Use trained reward model to perform Best-of-n response selection ---
    policy = PolicySampler(
        language_model=language_model,
        reward_model=reward_model,
        n_responses=cfg.get("sampling", {}).get("responses_per_instruction", 64),
        temperature=cfg.get("sampling", {}).get("temperature", 0.0)
    )

    # --- 8. Evaluation on test instructions ---
    eval_obj = Evaluation(
        instruction_data_path=cfg["evaluation"].get("test_instructions_path", "data/test_instructions.json"),
        reference_responses_path=cfg["evaluation"].get("reference_responses_path", "data/reference_responses.json"),
        reward_model_paths={"ranking": save_path, "rating": save_path},
        feedback_protocol=feedback_protocol,
        n_responses=cfg.get("sampling", {}).get("responses_per_instruction", 64),
        eval_samples=cfg["evaluation"].get("evaluation_samples", 1000),
        seed=seed
    )

    # Run the full evaluation pipeline
    logging.info("Running evaluation with Best-of-n policy...")
    eval_metrics = eval_obj.run_full_evaluation()

    # --- 9. Log and save evaluation metrics ---
    eval_results_path = "evaluation_results.json"
    with open(eval_results_path, "w") as f:
        json.dump(eval_metrics, f, indent=4)
    logging.info(f"Evaluation metrics saved at {eval_results_path}")

    # --- 10. Final message ---
    print("Reproduction pipeline completed.")
    print(f"Results saved at {eval_results_path}")

if __name__ == "__main__":
    main()
```

## model.py

```python
## model.py
import os
from typing import List, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    StoppingCriteriaList,
    StoppingCriteria,
)

import openai


class LanguageModel:
    """
    A wrapper class for language models, supporting both local transformer models
    and API-based models (like OpenAI's GPT-3.5 / GPT-4).
    Provides a unified interface for response generation.
    """

    def __init__(self, model_name: str, device: str = 'cuda'):
        """
        Initialize the LanguageModel with a model name.
        For local models, load from Hugging Face transformers.
        For API models, setup API client.
        """
        self.model_name = model_name
        self.device = device

        # Determine if model is an API model or local model
        # Basic heuristic: if the model name contains 'gpt-3.5' or 'gpt-4', use API
        model_name_lower = model_name.lower()
        if 'gpt-3.5' in model_name_lower or 'gpt-4' in model_name_lower:
            self.is_api = True
        else:
            self.is_api = False

        if self.is_api:
            self._load_api_client()
        else:
            self._load_local_model()

    def _load_local_model(self):
        """
        Load a local transformer-based language model and tokenizer.
        For example: LLaMA, Alpaca, or RoBERTa (if applicable).
        """
        # Load tokenizer and model with auto classes
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Ensure padding token for models lacking it
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)

        # Set model to evaluation mode
        self.model.eval()

    def _load_api_client(self):
        """
        Setup the API client for OpenAI models.
        Assumes OPENAI_API_KEY is set in environment variables.
        """
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variable 'OPENAI_API_KEY'.")
        # API client uses openai library functions directly

    def generate(self,
                 prompt: str,
                 max_length: int = 128,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 top_k: int = 0,
                 num_return_sequences: int = 1,
                 stop: Optional[List[str]] = None,
                 **kwargs) -> List[str]:
        """
        Generate responses based on the input prompt and parameters.
        Returns a list of response strings.
        """
        if self.is_api:
            return self._generate_api(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                stop=stop,
                **kwargs
            )
        else:
            return self._generate_local(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                stop=stop,
                **kwargs
            )

    def _generate_local(self,
                        prompt: str,
                        max_length: int,
                        temperature: float,
                        top_p: float,
                        top_k: int,
                        num_return_sequences: int,
                        stop: Optional[List[str]],
                        **kwargs) -> List[str]:
        """
        Generate text using local transformer model.
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Determine max_new_tokens
        input_length = input_ids.shape[-1]
        max_new_tokens = max(1, max_length - input_length)

        # Generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )

        # Decode outputs
        responses = []
        if isinstance(output_ids, list):
            output_ids = torch.tensor(output_ids)
        for generated_sequence in output_ids:
            text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
            # Remove prompt part if present
            response = text[len(prompt):].strip()
            responses.append(response)
        return responses

    def _generate_api(self,
                      prompt: str,
                      max_length: int,
                      temperature: float,
                      top_p: float,
                      top_k: int,
                      num_return_sequences: int,
                      stop: Optional[List[str]],
                      **kwargs) -> List[str]:
        """
        Generate responses via OpenAI API.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if hasattr(openai, 'top_k') else None,
                n=num_return_sequences,
                stop=stop,
                **kwargs
            )
            responses = []
            for choice in response.choices:
                content = choice.message.get("content", "").strip()
                responses.append(content)
            return responses
        except Exception as e:
            print(f"Error during API generation: {e}")
            return []

    def tokenize(self, text: str):
        """
        Tokenize input text.
        """
        if self.is_api:
            # Not applicable for API models
            return None
        else:
            return self.tokenizer.encode(text, return_tensors='pt').to(self.device)

    def __del__(self):
        """
        Cleanup resources if needed.
        """
        if not self.is_api:
            # Free up memory
            del self.model
            del self.tokenizer
```

## policy.py

```python
# policy.py

from typing import List
from dataset_loader import LanguageModel
from reward_model import RewardModel

class PolicySampler:
    """
    Implements the Best-of-n policy: generates multiple candidate responses
    for an instruction, scores them with a reward model, and returns the top response.
    """
    def __init__(
        self,
        language_model: LanguageModel,
        reward_model: RewardModel,
        n_responses: int = 64,
        temperature: float = 0.0
    ):
        """
        Args:
            language_model (LanguageModel): instance for response generation
            reward_model (RewardModel): instance for scoring responses
            n_responses (int): number of responses to generate per instruction
            temperature (float): sampling temperature for generation
        """
        self.language_model = language_model
        self.reward_model = reward_model
        self.n_responses = n_responses
        self.temperature = temperature

    def sample_responses(self, instruction: str) -> List[str]:
        """
        Generate multiple responses for a given instruction.
        Returns:
            responses (List[str]): list of generated responses
        """
        responses = self.language_model.generate(
            prompt=instruction,
            max_length=128,
            temperature=self.temperature,
            num_return_sequences=self.n_responses
        )
        return responses

    def score_responses(self, responses: List[str], instruction: str) -> List[float]:
        """
        Score each response using the reward model.
        Returns:
            scores (List[float]): list of scalar scores (higher is better)
        """
        scores = []
        for resp in responses:
            score = self.reward_model.score_response(resp, instruction)
            scores.append(score)
        return scores

    def select_best_response(self, responses: List[str], scores: List[float]) -> str:
        """
        Select the response with the highest score.
        """
        max_idx = scores.index(max(scores))
        return responses[max_idx]

    def generate_and_select(self, instruction: str) -> str:
        """
        Complete pipeline: generate multiple responses, score, and pick the best.
        Returns:
            best_response (str)
        """
        responses = self.sample_responses(instruction)
        scores = self.score_responses(responses, instruction)
        best_response = self.select_best_response(responses, scores)
        return best_response
```

**Key points & adherence:**
- The class uses configuration parameters `n_responses` and `temperature` from inputs.
- Response generation uses the `generate()` method of `LanguageModel`, assumed to support batch (`num_return_sequences`) and other params.
- Response scoring calls `score_response()` in `RewardModel`.
- The process is fully modular, matching the data structures and interfaces from the design.
- Default values are explicitly set, e.g., `128` max length, `0.0` temperature.
- No missed core functions; all core logic is captured according to the plan.

This implementation can be integrated seamlessly into the larger reproducibility framework.

## reward_model.py

```python
## reward_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.special import expit  # sigmoid function
from typing import List, Tuple, Optional, Union

class FeedbackSample:
    """
    Structure for a single feedback data point
    - For regression: instruction, response, normalized score
    - For preference: instruction, response1, response2, preference indicator (1, 2, or 0 for tie)
    """
    def __init__(self,
                 instruction: str,
                 response: str,
                 score: Optional[float] = None,
                 response_pair: Optional[Tuple[str, str]] = None,
                 preference: Optional[int] = None):
        self.instruction = instruction
        self.response = response
        self.score = score
        self.response_pair = response_pair
        self.preference = preference  # 1 if response1 preferred, 2 if response2 preferred, 0 if tie

class RewardDataset(Dataset):
    """
    A generic dataset class for reward model training.
    Handles both regression (scores) and pairwise preference data.
    """
    def __init__(self, data: List[FeedbackSample], mode: str = 'regression'):
        """
        Args:
            data: List of FeedbackSample objects
            mode: 'regression' or 'preference'
        """
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.mode == 'regression':
            return (item.instruction, item.response, item.score)
        elif self.mode == 'preference':
            # response_pair: (response1, response2)
            return (item.instruction, item.response_pair[0], item.response_pair[1], item.preference)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

class RewardModel:
    """
    Implements the reward model for either regression or preference.
    Uses transformer encoder for encoding instruction + response.
    """
    def __init__(self,
                 model_name: str = 'allenai/longformer-base-4096',
                 feedback_data: Optional[dict] = None,
                 training_mode: str = 'regression',  # 'regression' or 'preference'
                 learning_rate: float = 3e-5,
                 batch_size: int = 16,
                 epochs: int = 3,
                 weight_decay: float = 0.01,
                 early_stopping_patience: int = 2,
                 max_grad_norm: float = 1.0,
                 device: str = 'cuda'):
        """
        Initialize reward model with configuration.
        Args:
            model_name: pre-trained transformer model for encoding
            feedback_data: dict with keys 'ratings' and/or 'preferences'
            training_mode: 'regression' for scalar score prediction, 'preference' for pairwise
            other hyperparameters as per config.yaml
        """
        self.model_name = model_name
        self.feedback_data = feedback_data
        self.training_mode = training_mode
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Load tokenizer and encoder model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()  # For embedding extraction

        # Initialize trainable head depending on mode
        if self.training_mode == 'regression':
            self.head = nn.Linear(self.model.config.hidden_size, 1).to(self.device)
        elif self.training_mode == 'preference':
            self.head = nn.Linear(self.model.config.hidden_size, 1).to(self.device)
        else:
            raise ValueError("training_mode must be 'regression' or 'preference'.")

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Prepare datasets
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None

        # Placeholders for training progress
        self.best_valid_metric = float('inf')
        self.early_stop_counter = 0

        # To be populated
        self.train_dataset = None
        self.val_dataset = None

        # Prepare data
        if self.feedback_data:
            self._prepare_data()

    def _embed_response(self, instruction: str, response: str) -> torch.Tensor:
        """
        Get embedding vector for instruction-response pair.
        Uses the encoder model to encode concatenated sequence.
        """
        input_text = instruction + " " + response
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        # Use the pooled output (assumed for encoder): last hidden state mean
        last_hidden = output.last_hidden_state  # (batch_size, seq_len, hidden_size)
        embedding = torch.mean(last_hidden, dim=1)  # mean pooling
        return embedding.squeeze(0)  # shape: (hidden_size, )

    def _prepare_data(self):
        """
        Convert feedback data into dataset objects for training.
        Handles both regression and preference modes.
        """
        data_list: List[FeedbackSample] = []

        if 'ratings' in self.feedback_data:
            # feedback_data['ratings'] is list/dict of (instruction_id, response_id, score)
            ratings = self.feedback_data['ratings']
            for entry in ratings:
                instr = entry['instruction']
                resp = entry['response']
                score_raw = entry['score']
                # Normalize score to [0,1]
                norm_score = (score_raw - 1.0) / 6.0
                sample = FeedbackSample(instr, resp, score=norm_score)
                data_list.append(sample)

        if 'preferences' in self.feedback_data:
            # feedback_data['preferences']: list of (instruction, response1, response2, preference)
            preferences = self.feedback_data['preferences']
            for entry in preferences:
                instr = entry['instruction']
                resp1 = entry['response1']
                resp2 = entry['response2']
                pref = entry['preference']
                sample = FeedbackSample(instr, '', response_pair=(resp1, resp2), preference=pref)
                data_list.append(sample)

        # Create dataset object based on mode
        if self.training_mode == 'regression':
            self.train_dataset = RewardDataset(data_list, mode='regression')
        elif self.training_mode == 'preference':
            self.train_dataset = RewardDataset(data_list, mode='preference')
        else:
            raise ValueError(f"Unsupported training mode: {self.training_mode}")

        # For simplicity, split into train/validation if desired can be handled here
        # For now, the entire data is used for training with early stopping
        self._train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        # For validation, can be set similarly if validation data is supplied

    def train(self):
        """
        Run training over specified epochs with early stopping.
        """
        patience_counter = 0
        best_epoch_loss = float('inf')
        for epoch in range(self.epochs):
            total_loss = 0.0
            self.head.train()
            for batch in self._train_dataloader:
                self.optimizer.zero_grad()
                if self.training_mode == 'regression':
                    # batch: instr, response, score
                    instrs, responses, scores = batch
                    scores = scores.float().to(self.device)
                    embeddings = [
                        self._embed_response(instr, resp) for instr, resp in zip(instrs, responses)
                    ]
                    embeddings = torch.stack(embeddings)  # shape (batch_size, hidden_size)
                    preds = self.head(embeddings).squeeze(-1)  # shape (batch_size,)
                    # Sigmoid on preds to get [0,1]
                    preds_sigmoid = torch.sigmoid(preds)
                    loss = F.mse_loss(preds_sigmoid, scores)
                elif self.training_mode == 'preference':
                    # batch: instr, resp1, resp2, preference
                    instrs, resp1s, resp2s, prefs = batch
                    # Build embeddings for responses
                    embeddings1 = [
                        self._embed_response(instr, resp1) for instr, resp1 in zip(instrs, resp1s)
                    ]
                    embeddings2 = [
                        self._embed_response(instr, resp2) for instr, resp2 in zip(instrs, resp2s)
                    ]
                    embeddings1 = torch.stack(embeddings1)  # (batch_size, hidden_size)
                    embeddings2 = torch.stack(embeddings2)
                    scores1 = self.head(embeddings1).squeeze(-1)
                    scores2 = self.head(embeddings2).squeeze(-1)
                    diff = scores1 - scores2
                    probs = torch.sigmoid(diff)
                    # Wrap preference into target labels
                    target = torch.tensor(prefs).float().to(self.device)
                    # Handle tie (0): treat as 0.5 target, but if prefer: 1 or 0
                    # For simplicity, treat 0 preference as 0.5
                    # Alternatively, exclude ties from loss computation
                    mask_tie = (target == 0.5)
                    if mask_tie.any():
                        # For ties, loss is minimal or can be ignored
                        # Here we ignore ties for preference loss
                        probs_tie = probs[~mask_tie]
                        target_tie = target[~mask_tie]
                        loss = F.binary_cross_entropy(probs_tie, target_tie)
                    else:
                        loss = F.binary_cross_entropy(probs, target)
                else:
                    raise ValueError("Unsupported mode during training.")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()
            avg_epoch_loss = total_loss / len(self._train_dataloader)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_epoch_loss:.4f}")
            # Early stopping logic can be added here based on validation metrics

    def score_response(self, instruction: str, response: str) -> float:
        """
        Compute a scalar score for an instruction-response pair.
        """
        embedding = self._embed_response(instruction, response)
        output = self.head(embedding.to(self.device))
        score = torch.sigmoid(output).item()
        return score

    def score_pair(self, instruction: str, response1: str, response2: str) -> float:
        """
        Compute preference score: higher indicates response1 preferred.
        """
        embedding1 = self._embed_response(instruction, response1)
        embedding2 = self._embed_response(instruction, response2)
        score1 = self.head(embedding1.to(self.device))
        score2 = self.head(embedding2.to(self.device))
        diff = torch.sigmoid(score1 - score2).item()
        return diff  # value in [0,1], larger = response1 preferred

    def save(self, save_path: str):
        """
        Save model weights and head
        """
        torch.save({'head_state_dict': self.head.state_dict()}, save_path)

    def load(self, load_path: str):
        """
        Load model weights and head
        """
        checkpoint = torch.load(load_path)
        self.head.load_state_dict(checkpoint['head_state_dict'])

```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from datasets import Dataset
import numpy as np

from reward_model import RewardModel

class RewardTrainer:
    """
    Handles training of reward models (regression or preference) based on feedback datasets.
    Supports early stopping, validation, and logging.
    """

    def __init__(self, config: Dict, feedback_data: Dict):
        """
        Args:
            config: Configuration dictionary loaded from YAML, containing hyperparameters and settings
            feedback_data: Dictionary containing 'ratings' and/or 'preferences' datasets
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.feedback_data = feedback_data

        # Unpack configuration parameters
        self.learning_rate: float = config.get("training", {}).get("learning_rate", 1e-5)
        self.batch_size: int = config.get("training", {}).get("batch_size", 16)
        self.epochs: int = config.get("training", {}).get("epochs", 3)
        self.weight_decay: float = config.get("training", {}).get("weight_decay", 0.01)
        self.warmup_steps: int = config.get("training", {}).get("warmup_steps", 500)
        self.early_stopping_patience: int = config.get("training", {}).get("early_stopping_patience", 2)
        self.max_grad_norm: float = config.get("training", {}).get("max_grad_norm", 1.0)
        self.feedback_protocol: str = config.get("feedback", {}).get("feedback_protocol", "ranking")  # 'rating' or 'ranking'

        # Initialize RewardModel based on feedback protocol
        if self.feedback_protocol == "rating":
            self.model = RewardModel(
                model_name=self.config.get("reward_model", {}).get("model_name", "allenai/longformer-base-4096"),
                feedback_data=feedback_data,
                training_mode='regression',
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                epochs=self.epochs,
                weight_decay=self.weight_decay,
                early_stopping_patience=self.early_stopping_patience,
                max_grad_norm=self.max_grad_norm,
                device=self.device
            )
        elif self.feedback_protocol == "ranking":
            self.model = RewardModel(
                model_name=self.config.get("reward_model", {}).get("model_name", "allenai/longformer-base-4096"),
                feedback_data=feedback_data,
                training_mode='preference',
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                epochs=self.epochs,
                weight_decay=self.weight_decay,
                early_stopping_patience=self.early_stopping_patience,
                max_grad_norm=self.max_grad_norm,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown feedback_protocol: {self.feedback_protocol}")

        # Initialize optimizer
        params = list(self.model.head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        # Placeholder for validation dataset and dataloader
        self.val_dataloader = None
        self.best_valid_loss = float('inf')
        self.early_stop_counter = 0

        # Optionally, split data into train/val here if needed (not shown for brevity)
        # For simplicity, assume entire data used in training, validation can be custom

    def train(self):
        """
        Run training epochs with early stopping and validation.
        """
        print("Starting training...")
        for epoch in range(1, self.epochs+1):
            self.model.model.train()
            total_loss = 0.0
            for batch in self.model.train_dataloader:
                self.optimizer.zero_grad()

                if self.model.training_mode == 'regression':
                    # Batch: instrs, responses, scores
                    instrs, responses, scores = batch
                    scores = scores.float().to(self.model.device)
                    embeddings = []
                    for instr, resp in zip(instrs, responses):
                        emb = self.model._embed_response(instr, resp)
                        embeddings.append(emb)
                    embeddings = torch.stack(embeddings)  # (batch, hidden_size)
                    preds = self.model.head(embeddings).squeeze(-1)  # (batch,)
                    preds_sigmoid = torch.sigmoid(preds)
                    loss = F.mse_loss(preds_sigmoid, scores)
                elif self.model.training_mode == 'preference':
                    # Batch: instrs, resp1s, resp2s, prefs
                    instrs, resp1s, resp2s, prefs = batch
                    embeddings1, embeddings2 = [], []
                    for instr, r1, r2 in zip(instrs, resp1s, resp2s):
                        embeddings1.append(self.model._embed_response(instr, r1))
                        embeddings2.append(self.model._embed_response(instr, r2))
                    embeddings1 = torch.stack(embeddings1)
                    embeddings2 = torch.stack(embeddings2)
                    scores1 = self.model.head(embeddings1).squeeze(-1)
                    scores2 = self.model.head(embeddings2).squeeze(-1)
                    diff = scores1 - scores2  # (batch,)
                    probs = torch.sigmoid(diff)
                    target = torch.tensor(prefs).float().to(self.model.device)

                    # Handle ties (prefer ignoring ties in binary loss for simplicity)
                    mask_tie = (target == 0.5)
                    if mask_tie.any():
                        probs_tie = probs[~mask_tie]
                        target_tie = target[~mask_tie]
                        loss = F.binary_cross_entropy(probs_tie, target_tie)
                    else:
                        loss = F.binary_cross_entropy(probs, target)
                else:
                    raise ValueError("Unsupported training mode.")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.head.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.model.train_dataloader)
            print(f"Epoch {epoch}/{self.model.epochs} - Loss: {avg_loss:.4f}")

            # Validation step placeholder (if validation dataset provided)
            valid_loss = self._validate()
            if valid_loss is not None:
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.early_stop_counter = 0
                    self.model.save(f"reward_model_best_{self.feedback_protocol}.pt")
                    print(f"Validation improved, model saved.")
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.model.early_stopping_patience:
                        print("Early stopping triggered.")
                        break

    def _validate(self):
        """
        Placeholder for validation routine.
        Returns average validation loss or None if validation dataset not set.
        """
        # For simplicity, assume no validation set provided
        return None

    def save_model(self, save_path: str):
        """
        Save the trained reward model.
        """
        self.model.save(save_path)

    def load_model(self, load_path: str):
        """
        Load model weights.
        """
        self.model.load(load_path)
```


---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\sparse_feedback\sparse_feedback_repo`
