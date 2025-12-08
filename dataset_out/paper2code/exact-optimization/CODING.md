# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
# dataset_loader.py

import os
import json
import csv
import random
from typing import List, Tuple, Optional

class DatasetLoader:
    """
    Handles loading preference pairs and prompts for training and evaluation.
    Supports both synthetic/preference data formats and prompt-only datasets.
    """

    def __init__(self, dataset_path: str):
        """
        Initializes the DatasetLoader by loading preference pairs and prompts.
        Args:
            dataset_path (str): Directory or file prefix containing datasets.
                Expected to contain preference pairs and prompts files.
        """
        self.dataset_path = dataset_path
        self.preference_pairs: List[Tuple[str, str, str]] = []  # (prompt, response_winner, response_loser)
        self.prompts: List[str] = []

        # Load datasets
        self._load_preference_pairs()
        self._load_prompts()

    def _load_preference_pairs(self):
        """
        Loads preference pair data from a JSON or CSV file.
        The dataset should be structured as a list of samples:
        For JSON:
            [{"prompt": "...", "response_a": "...", "response_b": "...", "prefer": 1}, ...]
        For CSV:
            prompt,response_a,response_b,prefer
        """
        # Possible files
        json_path = os.path.join(self.dataset_path, "preference_pairs.json")
        csv_path = os.path.join(self.dataset_path, "preference_pairs.csv")

        if os.path.isfile(json_path):
            self.preference_pairs = self._load_from_json(json_path)
        elif os.path.isfile(csv_path):
            self.preference_pairs = self._load_from_csv(csv_path)
        else:
            # If no dataset found, issue warning and leave list empty
            print(f"Warning: Preference dataset not found in {self.dataset_path}")
            self.preference_pairs = []

    def _load_from_json(self, path: str) -> List[Tuple[str, str, str]]:
        """
        Loads preference pairs from a JSON file.
        """
        data = []
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)
            for sample in samples:
                prompt = sample.get("prompt", "").strip()
                response_a = sample.get("response_a", "").strip()
                response_b = sample.get("response_b", "").strip()
                prefer = sample.get("prefer", 0)  # 1 if response_a preferred, 2 if response_b preferred
                if prompt and response_a and response_b and prefer in [1, 2]:
                    # Convert preference to binary label: 1 if response_a preferred, 0 if response_b preferred
                    label = 1 if prefer == 1 else 0
                    # Store as (prompt, response_winner, response_loser)
                    if label == 1:
                        data.append((prompt, response_a, response_b))
                    else:
                        data.append((prompt, response_b, response_a))
        return data

    def _load_from_csv(self, path: str) -> List[Tuple[str, str, str]]:
        """
        Loads preference pairs from a CSV file.
        Each line: prompt,response_a,response_b,prefer
        """
        data = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                response_a = row.get("response_a", "").strip()
                response_b = row.get("response_b", "").strip()
                prefer_col = row.get("prefer", "").strip()
                if prompt and response_a and response_b and prefer_col:
                    prefer = int(prefer_col)
                    if prefer in [1, 2]:
                        label = 1 if prefer == 1 else 0
                        if label == 1:
                            data.append((prompt, response_a, response_b))
                        else:
                            data.append((prompt, response_b, response_a))
        return data

    def load_prompts(self, prompts_file: Optional[str] = None):
        """
        Loads prompts for evaluation from a text file.
        Args:
            prompts_file (str): Path to prompts file; defaults to 'prompts.txt' inside dataset_path.
        """
        if prompts_file is None:
            prompts_file = os.path.join(self.dataset_path, "prompts.txt")
        if os.path.isfile(prompts_file):
            with open(prompts_file, "r", encoding="utf-8") as f:
                self.prompts = [line.strip() for line in f if line.strip()]
        else:
            print(f"Warning: Prompts file not found at {prompts_file}")
            self.prompts = []

    def get_preference_batch(self, batch_size: int) -> Tuple[List[str], List[str], List[str], List[int]]:
        """
        Samples a batch of preference pairs for training.
        Returns:
            prompts: List of prompts
            responses_a: List of responses corresponding to response_winner
            responses_b: List of responses corresponding to response_loser
            labels: List[int], 1 if response_a preferred, 0 if response_b preferred
        """
        batch = random.choices(self.preference_pairs, k=batch_size)
        prompts, responses_a, responses_b, labels = [], [], [], []
        for prompt, resp_winner, resp_loser in batch:
            prompts.append(prompt)
            responses_a.append(resp_winner)
            responses_b.append(resp_loser)
            # Label: 1 if resp_winner was response_a, else 0
            labels.append(1)
        return prompts, responses_a, responses_b, labels

    def get_prompts(self) -> List[str]:
        """
        Returns the list of loaded prompts for evaluation.
        """
        return self.prompts
```

## evaluation.py

```python
## evaluation.py
import torch
import numpy as np
from typing import List, Dict
from utils import importance_weights, estimate_reverse_kl
from tqdm import tqdm

class Evaluator:
    """
    Implements evaluation routines for a trained language model (LM) policy and its divergence to 
    target or optimal policies, based on responses and reward scores.
    Provides methods to evaluate on datasets, estimate divergence (like reverse KL),
    and collect metrics for analysis consistent with the paper's evaluation protocols.
    """

    def __init__(
        self,
        lm,                   # Instance of LanguageModel
        reward_model,         # Instance of RewardModel
        eval_batch_size: int = 16,   # Batch size during evaluation
        divergence_method: str = "importance_sampling"  # Method for divergence estimation
    ):
        """
        Initialize the evaluator with models and evaluation settings.
        Args:
            lm (LanguageModel): Model for response generation and log probability estimation.
            reward_model (RewardModel): Model for scoring responses.
            eval_batch_size (int): Batch size for generation during evaluation.
            divergence_method (str): Method to estimate divergence ("importance_sampling" or "density_ratio").
        """
        self.lm = lm
        self.reward_model = reward_model
        self.eval_batch_size = eval_batch_size
        self.divergence_method = divergence_method

    def evaluate_on_dataset(self, prompts: List[str], responses: List[str]) -> Dict[str, float]:
        """
        Evaluate responses over a dataset, compute average reward scores,
        and other metrics as needed.
        Args:
            prompts (List[str]): List of prompts.
            responses (List[str]): Corresponding responses.
        Returns:
            Dict[str, float]: Metrics such as mean reward, std, response scores.
        """
        total_score = 0.0
        scores_list = []

        # Evaluate each prompt-response pair
        for prompt, response in tqdm(zip(prompts, responses), desc="Evaluating responses", leave=False):
            score = self.reward_model.score_response(prompt, response)
            scores_list.append(score)
            total_score += score

        avg_score = total_score / len(scores_list) if scores_list else 0.0

        metrics = {
            "average_reward": avg_score,
            "response_scores": scores_list,
        }
        return metrics

    def estimate_reverse_kl(self, prompts: List[str], responses: List[str], target_responses: List[List[str]]) -> float:
        """
        Estimate the reverse KL divergence D_KL(π_θ || π_β*) over the dataset.
        Args:
            prompts (List[str]): List of prompts.
            responses (List[str]): Responses generated by current policy for prompts.
            target_responses (List[List[str]]): Responses representing target/optimal policy distribution.
        Returns:
            float: Estimated divergence value.
        """
        # For each prompt, estimate density ratio using responses from current policy
        divergence_estimates = []

        for prompt, y_responses in tqdm(zip(prompts, responses), desc="Estimating divergence", leave=False):
            # For divergence estimation, responses are sampled from π_θ
            # Assume responses are generated, and for each response, we can estimate its likelihood ratio.
            # Here, we use importance sampling: estimate the ratio via responses' probabilities.

            # Compute response probabilities (or log probs) with model
            log_probs = self.lm.log_probs(prompt, y_responses)  # total log prob per response
            # Retrieve scaled reward scores for responses as in Eq. (23)
            # For divergence, assume target policy scores are known or estimated
            # Here, approximate as:
            # - Responses are responses[i], target density approx by pi_beta_star if available
            # - For simplicity, set target density as uniform or from the response scores.
            
            # Simplify: estimate density ratio as exp(f_theta(y)) / p_{target}(y)
            # So, compute importance weights (from utils): may need external density ratio estimator
            # For illustration, we'll assume the responses are from π_θ and target probabilities are known via scores.
            # Therefore, estimate divergence as mean of log density ratios.

            # If target response probabilities are available (e.g., from response scores under pi_beta*),
            # here, we simulate as proportional to exp(r_phi / some_scale), otherwise, as a placeholder:
            # using the response reward scores scaled as log probabilities
            # (assuming 'responses' are responses sampled from current policy, and target policy is estimated via response scores)
            # As divergence estimation is complex, here we just approximate as:
            # divergence per prompt:
            # For simplicity, use the log of response likelihood under the learned policy minus response scores as an approximation.
            # In practice, you can run density ratio estimators or importance sampling based on response likelihoods.
            # For the purpose of code structure, we will calculate:
            # divergence ≈ mean over responses of (log p(y|π_θ) - log p(y|π_β*))
            # using model log probs and response scores (scaling appropriately).

            # As a placeholder, we approximate divergence as:
            # difference between average log_probs and reward (assuming reward is log p under target)
            divergence = np.mean(log_probs)  # placeholder for actual divergence estimate
            divergence_estimates.append(divergence)

        # Return average divergence
        return float(np.mean(divergence_estimates))

    def compute_divergence_for_prompts(self, prompts: List[str]) -> Dict[str, float]:
        """
        For a list of prompts, generate responses with current policy
        and estimate divergence w.r.t. target (or previous policy).
        Args:
            prompts (List[str]): List of prompts.
        Returns:
            Dict[str, float]: Mean divergence estimate.
        """
        responses = []
        for prompt in tqdm(prompts, desc="Sampling responses for divergence", leave=False):
            # Generate responses with current policy
            resp_list = self.lm.generate_responses(prompt, num_responses=1, temperature=0.8)
            responses.append(resp_list[0])

        # For divergence estimation, assuming we have target responses or model
        # For the in-code placeholder, we can use the responses themselves as the target (not ideal)
        # or pass in target responses if available. Here, we proceed with responses as target.
        divergence_value = self.estimate_reverse_kl(prompts, responses, responses)
        return {"divergence": divergence_value}

    def detailed_evaluation(self, prompts: List[str]) -> Dict[str, float]:
        """
        Run complete evaluation: generate responses, score responses, estimate divergence.
        Args:
            prompts (List[str]): List of prompts.
        Returns:
            Dict[str, float]: Evaluation metrics including reward scores and divergence.
        """
        responses = []
        reward_scores = []

        for prompt in tqdm(prompts, desc="Generating responses", leave=False):
            resps = self.lm.generate_responses(prompt, num_responses=1, temperature=0.8)
            responses.append(resps[0])

        # Score responses
        for prompt, response in zip(prompts, responses):
            score = self.reward_model.score_response(prompt, response)
            reward_scores.append(score)

        # Estimate divergence
        divergence = self.estimate_reverse_kl(prompts, responses, responses)

        metrics = {
            "average_response_score": np.mean(reward_scores),
            "divergence": divergence,
            "responses": responses,
            "reward_scores": reward_scores
        }
        return metrics
```

## main.py

```python
#!/usr/bin/env python3
# main.py

import yaml
import os
import sys
import torch
import random
import numpy as np
from dataset_loader import DatasetLoader
from model import LanguageModel
from reward_model import RewardModel
from trainer import ModelTrainer
from evaluation import Evaluator
from utils import utils
from tqdm import tqdm

def main():
    # Load configuration
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = config.get('model', {}).get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        device = 'cpu'

    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # Paths and hyperparameters from config
    pretrained_model_path = config.get('model', {}).get('pretrained_model_path', 'gpt2-large')
    reward_model_path = config.get('reward_model', {}).get('model_path', '')
    preference_data_path = config.get('dataset', {}).get('preference_data_path', '')
    prompts_path = config.get('dataset', {}).get('prompts_path', '')
    max_training_steps = config.get('training', {}).get('max_training_steps', 5000)
    batch_size = config.get('training', {}).get('batch_size', 64)
    response_samples = config.get('training', {}).get('response_samples', 8)
    response_temperature = config.get('training', {}).get('response_temperature', 0.8)
    importance_beta = config.get('training', {}).get('importance_sampling_beta', 0.5)
    divergence_beta = config.get('training', {}).get('divergence_regularization_beta', 0.1)
    eval_steps = config.get('evaluation', {}).get('eval_steps', 3000)

    # Initialize dataset loader
    dataset_loader = DatasetLoader(preference_data_path)
    dataset_loader.load_prompts(prompts_path)

    # Initialize models
    lm = LanguageModel(pretrained_model_path, device=device)
    reward_model = RewardModel(reward_model_path, device=device)

    # Initialize trainer and evaluator
    trainer = ModelTrainer(lm, reward_model, dataset_loader, config)
    evaluator = Evaluator(lm, reward_model)

    # Optionally, set evaluator to the trainer if needed
    trainer.evaluator = evaluator

    # Main training loop
    print("Starting training...")
    for step in tqdm(range(1, max_training_steps + 1)):
        # Sample prompts for this batch
        prompts_batch = np.random.choice(dataset_loader.get_prompts(), size=batch_size, replace=True)

        responses_batch = []
        reward_scores_batch = []
        # Generate responses and score
        for prompt in prompts_batch:
            responses = lm.generate_responses(prompt, num_responses=response_samples, temperature=response_temperature)
            responses_batch.append(responses)
            scores = reward_model.score_responses(prompt, responses)
            reward_scores_batch.append(scores)

        # For each prompt, compute importance weights and update
        for prompt, responses, reward_scores in zip(prompts_batch, responses_batch, reward_scores_batch):
            # For each response, get log probs
            log_probs = lm.log_probs(prompt, responses)  # total log prob per response
            # Compute scaled reward scores for importance weights
            scaled_rewards = np.array(reward_scores) / importance_beta
            # Compute importance weights
            weights = utils.importance_weights(responses, scaled_rewards, None, importance_beta)  # logits placeholder
            # Perform a training step with weighted likelihood
            # The train_step function will handle gradient update
            trainer.train_step(prompt, responses, weights, divergence_beta)

        # Periodic evaluation
        if step % eval_steps == 0:
            print(f"Evaluation at step {step}")
            eval_prompts = dataset_loader.get_prompts()[:100]  # or pick a fixed eval set
            responses_eval = []
            for prompt in eval_prompts:
                responses = lm.generate_responses(prompt, num_responses=response_samples, temperature=response_temperature)
                responses_eval.append(responses)
            # Compute reward scores
            eval_rewards = []
            for prompt, responses in zip(eval_prompts, responses_eval):
                scores = reward_model.score_responses(prompt, responses)
                eval_rewards.extend(scores)
            # Estimate divergence (reverse KL)
            divergence_estimate = evaluator.estimate_reverse_kl(eval_prompts, responses_eval, None)  # placeholder
            # Log metrics
            avg_reward = np.mean(eval_rewards)
            print(f"Step {step}: Avg Reward = {avg_reward:.4f}, Divergence ~ {divergence_estimate:.4f}")
            # Optionally save checkpoint
            checkpoint_path = f'checkpoint_step_{step}.pt'
            trainer.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Final save
    final_path = 'final_model.pt'
    trainer.save_checkpoint(final_path)
    print(f"Training complete. Final model saved to {final_path}")

if __name__ == "__main__":
    main()
```

## model.py

```python
## model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional


class LanguageModel:
    """
    Defines the LanguageModel class, responsible for loading a pretrained language model,
    generating responses, and computing token-level log probabilities.
    It serves as the core component for sampling and likelihood estimation
    during training and evaluation, complying with the formalism and equations in the paper,
    particularly for response generation, logging, and divergence calculations.
    """

    def __init__(
        self,
        model_path: str = "gpt2-large",
        device: str = "cuda",
        max_response_length: int = 100,
        seed: Optional[int] = None
    ):
        """
        Initialize the language model:
        - Load the pretrained model and tokenizer
        - Set device (GPU/CPU)
        - Set maximum response token length for generation
        - Optionally set seed for reproducibility
        """
        self.model_path = model_path
        self.device = device
        self.max_response_length = max_response_length

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # Add BOS/EOS tokens if needed, depending on the model type
        # For GPT-2-like models, EOS token is sufficient
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = ""

        # Load pretrained model for causal language modeling
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()  # Set evaluation mode by default

        # For response generation, optionally set generation parameters
        self.generation_config = {
            "max_new_tokens": self.max_response_length,
            "temperature": 0.8,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "num_return_sequences": 1,
        }

    def set_eval(self):
        """Set model to evaluation mode."""
        self.model.eval()

    def set_train(self):
        """Set model to training mode."""
        self.model.train()

    def generate_responses(
        self,
        prompt: str,
        num_responses: int = 1,
        temperature: float = 0.8
    ) -> List[str]:
        """
        Generate a list of responses given a prompt.
        Args:
            prompt (str): The input prompt string.
            num_responses (int): Number of responses to generate.
            temperature (float): Sampling temperature.
        Returns:
            responses (List[str]): Generated response strings.
        """
        # Encode the prompt
        encoded_prompt = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Prepare generation config
        generation_kwargs = {
            "input_ids": encoded_prompt,
            "max_new_tokens": self.max_response_length,
            "temperature": temperature,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "num_return_sequences": num_responses,
        }

        # Generate responses
        try:
            generated_outputs = self.model.generate(**generation_kwargs)
        except Exception as e:
            print(f"Error during generation: {e}")
            return [""] * num_responses

        # Decode responses
        responses = []
        for output in generated_outputs:
            # Decode excluding prompt tokens (response only)
            # Find the position where prompt ends
            # Use decode for each output: exclude prompt tokens if needed
            decoded_text = self.tokenizer.decode(output, skip_special_tokens=True)
            responses.append(decoded_text.strip())

        return responses

    def log_probs(
        self,
        prompt: str,
        responses: List[str]
    ) -> List[float]:
        """
        Compute token-level log probabilities for each response conditioned on the prompt.
        Returns total (sequence) log probability for each response.
        """
        log_prob_list = []

        # Encode the prompt once
        prompt_encoding = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        for response in responses:
            # Encode full input: prompt + response
            response_encoding = self.tokenizer.encode(response, add_special_tokens=False)
            input_ids = torch.cat([prompt_encoding[0], torch.tensor(response_encoding, device=self.device)])
            input_ids = input_ids.unsqueeze(0)  # batch size = 1

            # Forward pass to get logits
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits.squeeze(0)  # shape: [seq_len, vocab_size]

            # Compute log probabilities
            # For each token in response, get predicted logits
            # Tokens: response_encoding, start after prompt length
            start_idx = len(prompt_encoding[0])
            response_tokens = response_encoding

            # To prevent mismatch
            if len(response_tokens) == 0:
                log_prob_list.append(0.0)
                continue

            token_logits = logits[start_idx -1 : start_idx -1 + len(response_tokens)]  # align response tokens
            # For each token, get log probs
            log_probs_per_token = torch.nn.functional.log_softmax(token_logits, dim=-1)
            token_log_probs = []
            for idx, token_id in enumerate(response_tokens):
                token_log_prob = log_probs_per_token[idx, token_id]
                token_log_probs.append(token_log_prob.item())

            total_log_prob = sum(token_log_probs)
            log_prob_list.append(total_log_prob)

        return log_prob_list

    def get_response_log_probs(
        self,
        prompt: str,
        responses: List[str]
    ) -> List[float]:
        """
        Alternative method: For batch processing of responses to compute log probs.
        Implementation similar to 'log_probs' but optimized for batch if needed.
        """
        # For simplicity, reuse 'log_probs'
        return self.log_probs(prompt, responses)

    def forward(
        self,
        prompt: str,
        temperature: float = 0.8
    ) -> str:
        """
        Generate a single response for a prompt with specified temperature.
        Convenient wrapper around generate_responses for single sample.
        """
        responses = self.generate_responses(prompt, num_responses=1, temperature=temperature)
        if responses:
            return responses[0]
        else:
            return ""

    def compute_response_log_prob(
        self,
        prompt: str,
        response: str
    ) -> float:
        """
        Compute log probability of a single response for the prompt.
        """
        return self.log_probs(prompt, [response])[0]

    def extract_logits(
        self,
        prompt: str,
        responses: List[str]
    ) -> List[torch.Tensor]:
        """
        For each response, obtain response token logits conditioned on prompt.
        Useful for divergence estimation or importance sampling.
        """
        # Implementation optional; for divergence calculations, batch process
        # For brevity, not implemented here, but can be built similar to 'log_probs'
        raise NotImplementedError("Direct logits extraction not implemented; use 'log_probs' method.")
```

## reward_model.py

```python
## reward_model.py

import os
import torch
import torch.nn.functional as F
from typing import List, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RewardModel:
    """
    Defines the RewardModel class to load a pretrained reward model, evaluate response scores
    for individual and batch responses given prompts, and provide interfaces for training and evaluation.
    Uses a classification or regression head based on the pretrained model.
    """

    def __init__(self, model_path: str = "path/to/reward/model", device: str = "cuda"):
        """
        Initializes the RewardModel:
        - Loads the pretrained reward model architecture and weights from model_path.
        - Loads the corresponding tokenizer.
        - Sets the model to evaluation mode.
        - Moves the model to the specified device.
        Args:
            model_path (str): Path or identifier for the pretrained reward model.
            device (str): 'cuda' or 'cpu', defaults to 'cuda' if available.
        """
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Check if model is a sequence classification or regression head
        # For simplicity, assume sequence classification with single label output
        # Users can adapt if using a different architecture
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()  # Set to eval mode for inference

    def score_response(self, prompt: str, response: str) -> float:
        """
        Scores a single response for a given prompt.
        Args:
            prompt (str): The input prompt.
            response (str): The response to score.
        Returns:
            float: Scalar reward score, higher indicates better response.
        """
        # Concatenate prompt and response or format input as needed
        input_text = f"Prompt: {prompt}\nResponse: {response}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Assume the model outputs logits with shape [batch_size, num_labels]
            logits = outputs.logits.squeeze(0)  # shape: [num_labels]
            # For binary classifier, logits[1] or logits[0] depending on class label
            # Assuming positive class score is at index 1
            # Also, if regression head, the output might be a single scalar
            if logits.shape[0] == 1:
                # Regression case
                score = logits.item()
            else:
                # Classification case
                # Use sigmoid on the positive class or max logit as proxy
                score = torch.sigmoid(logits[1]).item() if logits.shape[0] >=2 else logits[0].item()
        return score

    def score_responses(self, prompt: str, responses: List[str]) -> List[float]:
        """
        Scores a batch of responses for the same prompt.
        Args:
            prompt (str): The prompt string.
            responses (List[str]): List of responses to score.
        Returns:
            List[float]: List of scalar scores corresponding to responses.
        """
        # Prepare batch inputs
        input_texts = [
            f"Prompt: {prompt}\nResponse: {resp}"
            for resp in responses
        ]
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_batch = outputs.logits  # shape: [batch_size, num_labels]

        scores = []
        for logits in logits_batch:
            if logits.shape[0] == 1:
                # Regression case
                score = logits.item()
            else:
                # Classification case, sigmoid on positive class
                score = torch.sigmoid(logits[1]).item() if logits.shape[0] >= 2 else logits[0].item()
            scores.append(score)
        return scores

    # Optional: method to score multiple responses and prompts
    def score_batch(self, prompts: List[str], responses_list: List[List[str]]) -> List[List[float]]:
        """
        Scores many prompts with multiple responses each.
        Args:
            prompts (List[str]): List of prompts.
            responses_list (List[List[str]]): List of responses for each prompt.
        Returns:
            List[List[float]]: List of scores for each prompt's responses.
        """
        all_scores = []
        for prompt, responses in zip(prompts, responses_list):
            scores = self.score_responses(prompt, responses)
            all_scores.append(scores)
        return all_scores
```

**Notes & Assumptions:**
- The code loads a pretrained sequence classification model (e.g., a BERT, RoBERTa, or a fine-tuned classifier) from `model_path`. If a regression head is used, the output is a scalar.
- For scoring, the prompt and response are combined into a single text input. The exact formatting can be adapted as per the reward model training setup.
- For batch scoring, responses are processed together for efficiency.
- The code sets the model to evaluation mode and moves it onto the appropriate device.
- The class provides flexibility for extending to other input formats or model types if needed.
- Hyperparameters such as `max_length` or response length can be tuned based on dataset specifics.

This class can now be imported and instantiated in the training or evaluation scripts to evaluate responses efficiently.

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import numpy as np
from utils import importance_weights, log_probabilities, importance_weights
from tqdm import tqdm

class ModelTrainer:
    """
    Implements the core training routine for EXO: the importance-weighted maximum likelihood
    of responses per prompt, guided by the theoretical gradient alignment with the reversed KL.
    Manages response sampling, importance weight computation, gradient updates, and periodic evaluation.
    """

    def __init__(
        self,
        lm,  # LanguageModel instance
        reward_model,  # RewardModel instance
        dataset_loader,  # DatasetLoader instance
        config: dict  # loaded from config.yaml
    ):
        self.lm = lm
        self.reward_model = reward_model
        self.dataset_loader = dataset_loader

        # Hyperparameters from config with defaults
        self.lr = config.get("training", {}).get("learning_rate", 1e-5)
        self.batch_size = config.get("training", {}).get("batch_size", 64)
        self.response_samples = config.get("training", {}).get("response_samples", 8)
        self.response_temperature = config.get("training", {}).get("response_temperature", 0.8)
        self.importance_beta = config.get("training", {}).get("importance_sampling_beta", 0.5)
        self.divergence_beta = config.get("training", {}).get("divergence_regularization_beta", 0.1)
        self.max_steps = config.get("training", {}).get("max_training_steps", 5000)
        self.eval_interval = config.get("evaluation", {}).get("eval_steps", 3000)

        # Initialize optimizer for the language model parameters
        self.optimizer = optim.Adam(self.lm.model.parameters(), lr=self.lr)

        # Storage for step count
        self.global_step = 0

        # For evaluation
        self.evaluator = None  # Will instantiate later if needed

    def train(self):
        """
        Main training loop: for each iteration, sample prompts, responses, compute weights,
        update model, and periodically evaluate.
        """
        prompts = self.dataset_loader.get_prompts()

        for step in range(1, self.max_steps + 1):
            self.global_step = step
            # Sample batch prompts
            batch_prompts = np.random.choice(prompts, size=self.batch_size, replace=True)

            # Collect responses per prompt
            all_responses = []
            all_response_logits = []  # Store logits for importance weights
            for prompt in batch_prompts:
                responses = self.lm.generate_responses(
                    prompt,
                    num_responses=self.response_samples,
                    temperature=self.response_temperature
                )
                all_responses.append(responses)

            # Compute reward scores
            reward_scores_list = []
            for prompt, responses in zip(batch_prompts, all_responses):
                scores = self.reward_model.score_responses(prompt, responses)
                reward_scores_list.append(scores)

            # Flatten responses and scores for weight computation
            # We process one prompt at a time
            total_loss = 0.0
            self.optimizer.zero_grad()

            # Accumulate gradients over batch
            for prompt, responses, scores in zip(batch_prompts, all_responses, reward_scores_list):
                # Get token IDs of responses
                # Compute log_probs for responses
                log_probs = self.lm.log_probs(prompt, responses)
                # Simulate response logits for importance weights (or recompute as needed)
                # For this implementation, assume we can get token logits via log_probs
                # (alternatively, store logits during generation if available)
                # For simplification, we'll skip recomputing logits and use log_probs directly.

                # Compute f_theta(y) as scaled reward
                f_theta_y = (1.0 / self.importance_beta) * np.array(scores)
                # Compute importance weights
                weights = importance_weights(responses, f_theta_y, None, self.importance_beta)

                # Compute log probabilities of responses (already available)
                # Convert responses to token IDs to get exact log probs if needed
                # For simplicity, assume log_probs are total log likelihoods per response
                log_p_resp = torch.tensor(log_probs, dtype=torch.float32, device='cpu')

                # Compute weighted log-likelihood
                weighted_log_likelihood = torch.dot(
                    torch.tensor(weights, dtype=torch.float32),
                    log_p_resp
                )

                # Accumulate negative for loss (maximize likelihood)
                loss = -weighted_log_likelihood
                loss.backward()
                total_loss += loss.item()

            # Step optimizer
            self.optimizer.step()

            # Periodic evaluation
            if self.global_step % self.eval_interval == 0:
                self.evaluate()

    def evaluate(self):
        """
        Perform evaluation: estimate divergence (reverse KL), reward, etc.
        Can be extended to include metrics, plotting, or saving checkpoints.
        """
        # This function is a placeholder illustrating the evaluation logic
        # For actual implementation, generate responses, estimate divergence, compute reward
        prompts = self.dataset_loader.get_prompts()

        # Sample responses from current policy
        responses_by_prompt = []
        for prompt in prompts:
            responses = self.lm.generate_responses(
                prompt,
                num_responses=self.response_samples,
                temperature=self.response_temperature
            )
            responses_by_prompt.append(responses)

        # For divergence estimation:
        # Sample responses from a reference policy if available (e.g., SFT)
        # Here, for simplicity, assume responses are from current policy and reference is known
        # Use importance sampling to estimate reverse KL (see utils.py)

        # Placeholder: Compute an average reward score over validation set
        total_reward = 0.0
        total_responses = 0
        for prompt, responses in zip(prompts, responses_by_prompt):
            scores = self.reward_model.score_responses(prompt, responses)
            total_reward += sum(scores)
            total_responses += len(scores)

        avg_reward = total_reward / max(1, total_responses)
        print(f"Step {self.global_step}: Avg Reward = {avg_reward:.4f}")

        # Optional: estimate divergence if reference responses available
        # divergence = utils.estimate_reverse_kl(responses, reference_responses, prompts)

    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.
        """
        torch.save(self.lm.model.state_dict(), path)

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        """
        self.lm.model.load_state_dict(torch.load(path))
```

## utils.py

```python
## utils.py
import torch
import numpy as np
from scipy.stats import gaussian_kde

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the softmax probabilities from logits in a numerically stable way.
    Args:
        logits (torch.Tensor): Logits tensor of shape [batch_size, ..., vocab_size]
    Returns:
        torch.Tensor: Probabilities tensor of same shape, sum over last dim equals 1.
    """
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    exp_logits = torch.exp(logits - max_logits)
    sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
    probabilities = exp_logits / sum_exp
    return probabilities

def importance_weights(
    responses: list,
    reward_scores: list,
    response_logits: torch.Tensor,
    beta: float
) -> torch.Tensor:
    """
    Calculate importance weights proportional to exp(f_theta(y)) / p_{r_phi}(y).
    f_theta(y): scaled reward score, and p_{r_phi}(y): reward model probability.
    Args:
        responses (list): Sampled responses.
        reward_scores (list): Reward scores or log scores for responses.
        response_logits (torch.Tensor): Logits for responses [K, seq_len, vocab].
        beta (float): Importance sampling scale from config ('importance_sampling_beta').
    Returns:
        torch.Tensor: Importance weights normalized over responses [K].
    """
    # Compute log p_theta(y) via token log probabilities
    # Recompute total log probs if not provided
    # For now, assume response_logits are provided; otherwise, compute using 'log_prob'.

    # f_theta(y) = (1 / beta) * r_phi(x, y) + log p_theta(y) - log p_{sft}(y|x) (if needed)
    # Here, for importance weights, response scoring is scaled by 1/beta, responses' reward scores are provided.
    # Use the reward scores directly (assuming they are already scaled appropriately or log-prob).
    # Let's assume 'reward_scores' are in log scale for numerical stability, otherwise convert as needed.
    log_rewards = torch.tensor(reward_scores, dtype=torch.float32, device=response_logits.device)
    # For current policy probability, compute the log probs from logits
    log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
    # Sum over tokens in responses
    # Assuming responses are batch of token IDs, get total log probs per response
    # But responses are strings, so we must compute log probs externally, or responses are already tokenized
    # To avoid confusion, let's assume responses are list of token IDs, or responses are responses with log_probs computed prior.
    # For safety, compute the response log probs using 'log_probs' if responses are token IDs.
    # For this utility, just output weights proportional to exp(log_rewards - log_p_response)
    # Here, to proceed generically, assume 'reward_scores' are log-rates scaled by 1/beta.

    # Compute importance weights: exp(log_reward - log_p_theta)
    # But since we only have log rewards, and p_theta, we need to compute p_theta or approximate weights.
    # For simplicity, let's assume 'reward_scores' are scaled logs of exp(r_phi / beta), so weights are proportional to exp(r_phi / beta).
    # Alternatively, approximate as exp(log_rewards / beta).
    # To avoid overusing assumptions, we'll use log_rewards directly.

    # If response_logits are not used, and only response scores are available:
    # Calculate weights as exp(log_rewards / beta)
    weights = torch.exp(log_rewards / beta)
    # Normalize weights to sum to 1
    weights = weights / torch.sum(weights + 1e-8)
    return weights

def calculate_density_ratio(
    responses: list,
    response_probs: np.ndarray,
    method: str = "kernel_density"
):
    """
    Estimate the density ratio pi(y|x) / pi_sft(y|x) using kernel density estimation
    or importance sampling based on provided response probabilities.
    Args:
        responses (list): Responses sampled from the policy.
        response_probs (np.ndarray): Probabilistic density estimates of responses.
        method (str): 'kernel_density' or 'importance_sampling'.
    Returns:
        np.ndarray: Estimated density ratios for each response.
    """
    if method == "kernel_density":
        # Use Gaussian KDE on responses (usually high dimensional),
        # response is high dim, so this is a simplification or placeholder.
        # In practice, responses are sequences; kernel density estimation is complex.
        # Here, assume responses are embedded or represented as vectors.
        # Placeholder: return uniform ratios or scaled responses.
        # For in-train visualization, actual embedding and density estimation should be used.
        # For now, just return the response_probs scaled as ratios.
        ratios = response_probs
    elif method == "importance_sampling":
        # Assume response_probs are importance weights or probabilities,
        # ratios = response_probs / pi_sft(y|x) estimates.
        ratios = response_probs
    else:
        raise ValueError(f"Unknown density estimation method: {method}")
    return ratios

def estimate_reverse_kl(
    responses: list,
    density_ratios: np.ndarray,
    prompts: list,
    method: str = "importance_sampling",
    divergence_type: str = "reverse_kl"
) -> float:
    """
    Estimate the divergence (reverse or forward KL) between policies using importance sampling.
    Args:
        responses (list): Responses sampled, corresponding to prompts.
        density_ratios (np.ndarray): Estimated ratios (pi / pi_sft or vice versa).
        prompts (list): List of prompts, to average over.
        method (str): Estimation method ('importance_sampling', 'density_ratio', 'kernel_density').
        divergence_type (str): 'reverse_kl' or 'forward_kl'.
    Returns:
        float: Estimated divergence value.
    """
    # Compute per-response contributions
    log_ratios = np.log(density_ratios + 1e-8)
    # For reverse KL: E_pi[log(pi / tilde_pi)] = sum over responses of pi * log ratio
    # Approximate expectation as mean over responses weighted by pi
    # Here, responses are sampled from pi, so simply average log ratios
    divergence_estimate = np.mean(log_ratios)
    return divergence_estimate

def log_probabilities(
    lm,  # LanguageModel object
    responses: list,
    prompt: str
) -> list:
    """
    Calculate total log probabilities of responses given prompt from the language model.
    Args:
        lm: Instance of LanguageModel with a 'log_probs' method.
        responses (list): Responses as strings.
        prompt (str): The associated prompt.
    Returns:
        list: Log probability for each response.
    """
    # Use the lm's log_probs method
    return lm.log_probs(prompt, responses)

def compute_tools():
    """
    Return utility functions or encapsulate as needed.
    """
    return {
        "softmax": softmax,
        "importance_weights": importance_weights,
        "calculate_density_ratio": calculate_density_ratio,
        "estimate_reverse_kl": estimate_reverse_kl,
        "log_probabilities": log_probabilities
    }
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\exact-optimization\exact-optimization_repo`
