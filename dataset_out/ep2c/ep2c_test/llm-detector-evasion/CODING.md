# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import random
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from detectors import DetectorAPI
from model import ModelWrapper
from utils import preference_probability, normalize_scores


class DatasetLoader:
    """
    Loads prompts, generates responses from models, obtains detector scores,
    and constructs preference pairs for RL training.
    """

    def __init__(
        self,
        prompts: Optional[List[str]] = None,
        responses: Optional[List[str]] = None,
        prompt_source: str = "file",
        prompt_file_path: str = "",
        detector: Optional[DetectorAPI] = None,
        max_responses_per_prompt: int = 2,
        generation_params: Dict = None,
        detector_score_scale: str = "log_prob",
        random_seed: int = 42,
    ):
        """
        Initialize the DatasetLoader.

        Args:
            prompts (Optional[List[str]]): List of prompt strings.
            responses (Optional[List[str]]): List of responses corresponding to prompts.
            prompt_source (str): Method to load prompts ("file", "dataset", etc.).
            prompt_file_path (str): Path to prompt file if prompt_source=="file".
            detector (DetectorAPI): The detector API/model instance.
            max_responses_per_prompt (int): Responses to generate per prompt.
            generation_params (Dict): Sampling parameters for generation.
            detector_score_scale (str): Scale of detector scores ("log_prob" or "prob").
            random_seed (int): Seed for reproducibility.
        """
        import numpy as np
        import os

        self.prompts = prompts or []
        self.responses = responses or []
        self.prompt_source = prompt_source
        self.prompt_file_path = prompt_file_path
        self.detector = detector
        self.max_responses_per_prompt = max_responses_per_prompt
        self.generation_params = generation_params or {
            "max_new_tokens": 120,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        self.detector_score_scale = detector_score_scale
        self.random_seed = random_seed
        self.preference_pairs: List[Tuple[str, str, str, int]] = []

        # Load prompts if needed
        if not self.prompts:
            self._load_prompts()

    def _load_prompts(self):
        """
        Load prompts from a file specified in configuration or set to empty.
        """
        if self.prompt_source == "file" and self.prompt_file_path:
            with open(self.prompt_file_path, "r", encoding="utf-8") as f:
                self.prompts = [line.strip() for line in f if line.strip()]
        else:
            # Placeholder for other loading methods, e.g., datasets
            self.prompts = []
        assert self.prompts, "Prompts list is empty after loading."

    def generate_responses(
        self,
        model: ModelWrapper,
        responses_per_prompt: int = 2,
        use_responses: Optional[List[str]] = None,
    ):
        """
        Generate responses for each prompt using the provided model.

        Args:
            model (ModelWrapper): Wrapped model for generation.
            responses_per_prompt (int): Number of responses to generate per prompt.
            use_responses (Optional[List[str]]): Optional existing responses to assign.

        Side effects:
            Populates self.responses with generated responses.
        """
        self.responses = []
        for prompt in tqdm(self.prompts, desc="Generating responses"):
            responses_for_prompt = []
            for _ in range(responses_per_prompt):
                response = model.generate(
                    prompt=prompt,
                    max_new_tokens=self.generation_params["max_new_tokens"],
                    temperature=self.generation_params["temperature"],
                    top_p=self.generation_params["top_p"]
                )
                responses_for_prompt.append(response.strip())
            self.responses.append(responses_for_prompt)

    def compute_detector_scores(self):
        """
        Obtain detector scores for each response.

        Returns:
            List[List[float]]: List of response scores for each prompt.
        """
        if self.detector is None:
            raise ValueError("Detector API/model not provided.")
        scores_per_prompt = []

        for responses in tqdm(self.responses, desc="Scoring responses"):
            scores = []
            for response in responses:
                score = self.detector.score(response)
                scores.append(score)
            scores_per_prompt.append(scores)
        return scores_per_prompt

    def create_preference_pairs(
        self,
        detector_scores: List[List[float]],
        threshold: float = 0.0,
    ):
        """
        For each prompt, compare responses based on detector scores and create preference pairs.

        Args:
            detector_scores (List[List[float]]): Scores for responses per prompt.
            threshold (float): Minimum score difference to assign preference.
        Side effects:
            Fills self.preference_pairs with tuples:
            (prompt, response_w, response_l, label)
            label=1 if response_w > response_l, else 0.
        """
        self.preference_pairs = []
        for idx, responses in enumerate(self.responses):
            scores = detector_scores[idx]
            # Generate all pairs (or pairwise comparisons)
            # Assuming responses list has at least 2 responses
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    score_i = scores[i]
                    score_j = scores[j]
                    # Use threshold to determine preference
                    if abs(score_i - score_j) < threshold:
                        # Difference too small; skip or assign randomly
                        continue
                    if score_i > score_j:
                        label = 1  # response i preferred
                        self.preference_pairs.append(
                            (self.prompts[idx], responses[i], responses[j], label)
                        )
                    else:
                        label = 1  # response j preferred
                        self.preference_pairs.append(
                            (self.prompts[idx], responses[j], responses[i], label)
                        )

    def get_pairs(self) -> List[Tuple[str, str, str, int]]:
        """
        Return the assembled preference pairs.

        Returns:
            List of tuples: (prompt, response_w, response_l, label)
        """
        return self.preference_pairs

    def sample_pairs(self, batch_size: int) -> List[Tuple[str, str, str, int]]:
        """
        Randomly sample a batch of preference pairs for training.

        Args:
            batch_size (int): Number of pairs to sample.

        Returns:
            List of sampled pairs.
        """
        if len(self.preference_pairs) == 0:
            raise ValueError("Preference pairs dataset is empty. Run create_preference_pairs() first.")
        return random.sample(self.preference_pairs, min(batch_size, len(self.preference_pairs)))
```

## detectors.py

```python
## detectors.py
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from typing import Union, List, Optional
import logging

logger = logging.getLogger(__name__)

class DetectorAPI:
    """
    A unified interface for different types of detectors:
    - API-based detector endpoints
    - Local open-source models
    """

    def __init__(self, model_type: str = "api", config: dict = None):
        """
        Initialize the DetectorAPI based on model_type.
        Args:
            model_type (str): "api" or "local"
            config (dict): Configuration parameters:
                For "api" type:
                    - api_endpoint (str): URL of the detector API
                    - api_token (str, optional): Auth token for API
                For "local" type:
                    - model_name (str): Huggingface model identifier
                    - device (str): "cuda" or "cpu"
        """
        self.model_type = model_type.lower()
        self.config = config or {}
        if self.model_type == "api":
            self.api_endpoint = self.config.get("api_endpoint", "")
            self.api_token = self.config.get("api_token", "")
            self.headers = {}
            if self.api_token:
                self.headers["Authorization"] = f"Bearer {self.api_token}"
        elif self.model_type == "local":
            self.model_name = self.config.get("model_name", "roberta-base")
            self.device = self.config.get("device", "cpu")
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def score(self, text: str) -> float:
        """
        Compute the detector score for a single text.
        Returns:
            float: scalar score where higher indicates "more human-like"
        """
        if self.model_type == "api":
            return self._score_api(text)
        elif self.model_type == "local":
            return self._score_model(text)
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}")

    def batch_score(self, texts: List[str]) -> List[float]:
        """
        (Optional) Batch scoring for efficiency.
        For API-based models, process sequentially.
        For local models, batch process.
        """
        scores = []
        if self.model_type == "api":
            for txt in texts:
                scores.append(self._score_api(txt))
        elif self.model_type == "local":
            # Batch encode
            encodings = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            with torch.no_grad():
                outputs = self.model(**encodings)
                # Assuming the model's logits correspond to classes: e.g., human vs ai
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                # Suppose 'human' label is class 0, 'AI' is class 1
                # Adjust accordingly if different
                # Here, assume higher probability for class 0 means more human-like
                # So, use probability of class 0
                if probs.shape[1] >= 2:
                    human_prob = probs[:, 0]
                else:
                    # If only one class, fallback to sigmoid of logit
                    human_prob = torch.sigmoid(logits.squeeze())
                scores.extend(human_prob.cpu().numpy().tolist())
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        return scores

    def _score_api(self, text: str) -> float:
        """
        Send a request to detector API and parse response.
        Expect the API to return a JSON with a score or probability.
        """
        payload = {"text": text}
        try:
            response = requests.post(self.api_endpoint, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            # Expect data to contain 'score' or 'probability'
            if "score" in data:
                score = float(data["score"])
            elif "probability" in data:
                score = float(data["probability"])
            elif "human_score" in data:
                score = float(data["human_score"])
            else:
                # fallback: try to fetch 'confidence' or similar
                # Default fallback if schema unknown
                score = float(data.get("confidence", 0.5))
            return score
        except requests.RequestException as e:
            # Log error; return neutral or conservative score
            logger.warning(f"API request failed: {e}. Returning default score 0.5")
            return 0.5  # neutral score
        except ValueError as ve:
            logger.warning(f"Failed to parse response: {ve}. Returning default score 0.5")
            return 0.5

    def _score_model(self, text: str) -> float:
        """
        Compute the log likelihood or probability of the text under the model,
        converting to a scalar score (e.g., probability of being human).
        """
        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape: (1, seq_len, vocab_size) or (1, seq_len, 2)
            # Compute negative sum of log probabilities for the sequence
            # Probability of entire sequence
            # Alternatively, get per-token probabilities and average
            # Here, assume binary classifier for detectors
            if logits.ndim == 2:
                # Single output per token
                probs = F.softmax(logits, dim=-1)
                # Use probability of the "human" class if binary
                if probs.shape[1] >= 2:
                    human_prob = probs[:, 0]
                else:
                    human_prob = torch.sigmoid(logits.squeeze())
                # Compute sum over tokens
                # Larger likelihood indicates more "human-like"
                ll = torch.sum(torch.log(human_prob + 1e-12))
                score = torch.exp(ll / len(input_ids[0]))  # normalized likelihood
            elif logits.ndim == 3:
                # For sequence classification head (e.g., RoBERTa), get logits for class
                # Here, assuming sequence classification
                probs = F.softmax(logits, dim=-1)
                # Take the mean probability over tokens or use the logits' pred
                # For simplicity, use logits for the sequence class if available
                # For sequence classifiers, logits shape: (batch_size, num_classes)
                # Extend to get full sequence
                sequence_logit = logits[:, -1, :]  # last token logits
                if sequence_logit.shape[1] >= 2:
                    human_prob = torch.sigmoid(sequence_logit[:, 0])  # assuming class 0 is human
                else:
                    human_prob = torch.sigmoid(sequence_logit.squeeze())
                score = human_prob.item()
            else:
                # fallback
                score = 0.5
        # Ensure score is within [0,1]
        score = max(0.0, min(score, 1.0))
        return score
```

## evaluation.py

```python
## evaluation.py
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import expit
from typing import List, Dict, Optional
from detectors import DetectorAPI
from model import ModelWrapper

class Evaluation:
    """
    Evaluation class to compute metrics such as AUROC and perplexity for a given
    set of texts, using specified detectors and a language model.
    """

    def __init__(
        self,
        model: ModelWrapper,
        detectors: List[DetectorAPI],
        device: str = "cuda",
        detector_names: Optional[List[str]] = None,
        human_texts: Optional[List[str]] = None,
        human_labels: Optional[List[int]] = None,
        texts: Optional[List[str]] = None,
        labels: Optional[List[int]] = None,
    ):
        """
        Initializes the evaluator.
        Args:
            model (ModelWrapper): The language model for perplexity eval.
            detectors (List[DetectorAPI]): List of detector instances.
            device (str): 'cuda' or 'cpu'.
            detector_names (Optional[List[str]]): Names corresponding to detectors.
            human_texts (Optional[List[str]]): Texts presumed human-written (for AUROC).
            human_labels (Optional[List[int]]): Labels for human texts (1).
            texts (Optional[List[str]]): Texts to evaluate for metrics.
            labels (Optional[List[int]]): True labels for texts, 1=human, 0=AI.
        """
        self.model = model
        self.detectors = detectors
        self.device = device
        self.detector_names = detector_names or [f"Detector_{i}" for i in range(len(detectors))]
        self.human_texts = human_texts
        self.human_labels = human_labels
        self.texts = texts
        self.labels = labels

        # Validate inputs
        if self.human_texts is not None and self.human_labels is None:
            raise ValueError("If human_texts is provided, human_labels must be provided.")
        if self.texts is not None and self.labels is None:
            raise ValueError("If texts are provided, labels must be provided.")

    def compute_detector_scores(
        self,
        texts: List[str],
    ) -> Dict[str, List[float]]:
        """
        Obtain scores from all detectors for each text.
        Args:
            texts (List[str]): Texts to score.
        Returns:
            Dict[str, List[float]]: A dict mapping detector name to list of scores.
        """
        scores_dict = {}
        for det, name in zip(self.detectors, self.detector_names):
            scores = []
            for text in texts:
                score = det.score(text)
                scores.append(score)
            scores_dict[name] = scores
        return scores_dict

    def evaluate_detector_auroc(
        self,
        detector_scores: Dict[str, List[float]],
        true_labels: List[int],
    ) -> Dict[str, float]:
        """
        Compute AUROC for each detector given scores and ground truth labels.
        Args:
            detector_scores (Dict[str, List[float]]): Scores per detector.
            true_labels (List[int]): True labels (1=human, 0=AI).
        Returns:
            Dict[str, float]: AUROC value per detector.
        """
        auroc_dict = {}
        for name in detector_scores:
            scores = detector_scores[name]
            try:
                auroc = roc_auc_score(true_labels, scores)
            except Exception:
                auroc = float('nan')
            auroc_dict[name] = auroc
        return auroc_dict

    def compute_perplexity(self, texts: List[str]) -> float:
        """
        Compute the average perplexity over a list of texts.
        Args:
            texts (List[str]): List of texts.
        Returns:
            float: Average perplexity.
        """
        total_neg_log_likelihood = 0.0
        total_tokens = 0
        for text in texts:
            # Use model's log_prob method
            ll = self.model.log_prob(text)
            tokens = self.model.tokenizer.tokenize(text)
            token_count = max(1, len(tokens))
            # Approximate negative log-likelihood
            neg_ll = -ll
            total_neg_log_likelihood += neg_ll
            total_tokens += token_count
        if total_tokens == 0:
            return float('nan')
        avg_neg_ll = total_neg_log_likelihood / total_tokens
        perplexity = np.exp(avg_neg_ll)
        return perplexity

    def evaluate_texts(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
    ) -> Dict:
        """
        Evaluate texts: compute detector scores, AUROC (if labels provided),
        and perplexity.
        Args:
            texts (List[str]): List of texts to evaluate.
            labels (Optional[List[int]]): Ground truth labels for AUROC.
        Returns:
            Dict: Dictionary with metrics.
        """
        scores_per_detector = self.compute_detector_scores(texts)
        result = {}
        # Compute detector scores and stats
        for name, scores in scores_per_detector.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            result[f"{name}_mean_score"] = mean_score
            result[f"{name}_std_score"] = std_score

        # Compute AUROC if labels are provided
        if labels is not None:
            for name, scores in scores_per_detector.items():
                try:
                    auroc = roc_auc_score(labels, scores)
                except Exception:
                    auroc = float('nan')
                result[f"AUROC_{name}"] = auroc

        # Compute perplexity
        perplexity = self.compute_perplexity(texts)
        result['perplexity'] = perplexity

        return result

    def human_preference_evaluation(
        self,
        responses_pairs: List[Tuple[str, str]],
        detector: DetectorAPI,
    ) -> Dict[str, float]:
        """
        Human evaluation based on detector scores of response pairs.
        Args:
            responses_pairs (List[Tuple[str, str]]): List of (response1, response2).
            detector (DetectorAPI): To score responses.
        Returns:
            Dict[str, float]: Proportion of responses where response1 is more human.
        """
        count_response1_preferred = 0
        total = len(responses_pairs)
        for r1, r2 in responses_pairs:
            score1 = detector.score(r1)
            score2 = detector.score(r2)
            # Higher detector score indicates more human-like
            if score1 > score2:
                count_response1_preferred += 1
        proportion = count_response1_preferred / total if total > 0 else 0.0
        return {
            "response1_better_proportion": proportion,
            "total_pairs": total
        }

    # Optional: add additional metrics or human annotation analysis as needed
```

## main.py

```python
## main.py
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm

from dataset_loader import DatasetLoader
from detectors import DetectorAPI
from model import ModelWrapper
from trainer import PPOTrainer
from evaluation import Evaluation
from utils import normalize_scores

def main():
    # 1. Load config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    seed = 42
    if "seed" in config:
        seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Initialize device
    device = config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Initialize DetectorAPI (assuming API-based here; customize in config if local)
    detector_config = config.get("detectors", {})
    detector_api = None
    if detector_config.get("api_endpoint", "") != "":
        detector_api = DetectorAPI(model_type="api",
                                   config={
                                       "api_endpoint": detector_config.get("api_endpoint", ""),
                                       "api_token": detector_config.get("api_token", "")
                                   })
    else:
        # For local model detector, modify accordingly
        detector_api = DetectorAPI(model_type="local",
                                   config={
                                       "model_name": detector_config.get("model_name", "roberta-base"),
                                       "device": device
                                   })

    # 4. Load prompt dataset
    prompts = []
    # Assuming openwebtext prompts: create a placeholder or load dataset
    # Here, we hardcode prompts or load from file if specified
    if "prompt_source" in config["dataset"]:
        source = config["dataset"]["prompt_source"]
        if source == "file" and "prompt_file_path" in config["dataset"]:
            path = config["dataset"]["prompt_file_path"]
            with open(path, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
        elif source == "openwebtext":
            # Placeholder: sample prompts from dataset or define static prompts
            # For reproducibility, define simple prompts
            prompts = [
                "The quick brown fox",
                "In a far away land",
                "Once upon a time",
                "The future of AI is",
                "The history of art involves",
                "Advancements in science",
                "The benefits of exercise",
                "Understanding quantum mechanics"
            ]
        else:
            prompts = ["Sample prompt 1", "Sample prompt 2"]
    else:
        prompts = ["Sample prompt 1", "Sample prompt 2"]

    # 5. Instantiate denormalized model
    model_name = config["training"].get("model_name", "Llama-2-7b")
    model_wrapper = ModelWrapper(model_name=model_name, device=device)

    # 6. Generate baseline responses
    print("Generating baseline responses...")
    dataset = DatasetLoader(prompts=prompts)
    dataset.generate_responses(
        model=model_wrapper,
        responses_per_prompt=2,
        responses=None  # Will generate new responses internally
    )

    # 7. Obtain detector scores for generated responses (initial responses)
    detector_scores = dataset.compute_detector_scores()
    # print("Sample detector scores:", detector_scores[:2])  # Optional

    # 8. Create preference dataset based on detector scores
    print("Constructing preference pairs based on detector scores...")
    dataset.create_preference_pairs(detector_scores, threshold=0.0)  # threshold=0.0 for strict comparison
    preference_pairs = dataset.get_pairs()
    print(f"Total preference pairs: {len(preference_pairs)}")

    # 9. Instantiate PPO trainer
    training_params = config["training"]
    ppo_trainer = PPOTrainer(
        model=model_wrapper,
        dataset=dataset,
        detector=detector_api,
        kl_coeff=training_params.get("kl_coeff", 0.5),
        beta=training_params.get("beta", 0.5),
        lr=training_params.get("learning_rate", 1e-5),
        batch_size=training_params.get("batch_size", 16),
        total_steps=training_params.get("total_steps", 30000),
        save_interval=training_params.get("save_interval", 5000),
        device=device,
        sequence_length=training_params.get("sequence_length", 120),
        temperature=training_params.get("temperature", 0.7),
        top_p=training_params.get("top_p", 0.9)
    )

    # 10. Run RL fine tuning
    print("Starting RL fine-tuning to evade detector...")
    ppo_trainer.train()

    # 11. Save final model
    final_model_path = "llama2-7b-evading.pt"
    print(f"Saving final fine-tuned model to {final_model_path}")
    torch.save(model_wrapper.model.state_dict(), final_model_path)

    # 12. Generate validation/evaluation samples after training
    eval_prompts = prompts  # or load separate evaluation prompts if desired
    eval_responses = []
    print("Generating eval responses...")
    for prompt in tqdm(eval_prompts, desc="Generating eval samples"):
        resp = model_wrapper.generate(
            prompt=prompt,
            max_new_tokens=training_params.get("sequence_length", 120),
            temperature=training_params.get("temperature", 0.7),
            top_p=training_params.get("top_p", 0.9)
        )
        eval_responses.append((prompt, resp))

    # 13. Evaluate detector scores, AUROC, perplexity
    print("Evaluating on generated samples...")
    detector_scores_eval = {}
    for detector_name in [detectors.model_type for detectors in [detector_api]]:
        # get scores
        scores = []
        for _, resp in eval_responses:
            scores.append(detector_api.score(resp))
        detector_scores_eval[detector_name] = scores

    # Or use the evaluation.py module for a comprehensive report
    eval_metrics = Evaluation(
        model=model_wrapper,
        detectors=[detector_api],
        device=device
    ).evaluate_texts([resp for _, resp in eval_responses])

    print("Evaluation metrics after fine-tuning:")
    print(eval_metrics)

    # 14. Optional: Human evaluation
    # Can be implemented by an external survey or omitted here

if __name__ == "__main__":
    main()
```

## model.py

```python
## model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional

class ModelWrapper:
    """
    A wrapper around Hugging Face transformer models for sequence generation,
    log probability computation, and detector score retrieval.
    """

    def __init__(self, model_name: str = "facebook/llama-2-7b-chat", device: str = "cuda"):
        """
        Initialize the ModelWrapper by loading the specified model and tokenizer.

        Args:
            model_name (str): Pretrained model identifier on Hugging Face.
            device (str): 'cuda' or 'cpu'.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.model_name = model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Set padding token if not set to avoid errors
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 120, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate a continuation for the given prompt.

        Args:
            prompt (str): Input prompt string.
            max_new_tokens (int): Max tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling probability threshold.

        Returns:
            str: Generated text.
        """
        # Encode prompt
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)

        # Generate response using model.generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode output
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Remove prompt part from generated output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text

    def log_prob(self, sequence: str, input_prompt: str) -> float:
        """
        Compute the log probability of sequence given input prompt.

        Args:
            sequence (str): The full sequence (prompt + response).
            input_prompt (str): The prompt string used to generate sequence.

        Returns:
            float: Log probability scalar.
        """
        # Tokenize prompt and sequence
        prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(self.device)
        seq_ids = self.tokenizer.encode(sequence, return_tensors='pt').to(self.device)

        # Concatenate prompt and sequence ids
        input_ids = torch.cat([prompt_ids, seq_ids], dim=1)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits  # shape: (1, total_seq_len, vocab_size)

        # Compute log probabilities
        # For each token in sequence (excluding prompt), compute probability conditioned on previous tokens
        # The prompt tokens are at the start; start index = prompt length
        prompt_len = prompt_ids.shape[1]
        seq_len = seq_ids.shape[1]
        total_log_prob = 0.0

        for i in range(prompt_len, prompt_len + seq_len):
            # The context is all tokens before token i
            input_ids_slice = input_ids[:, :i]
            with torch.no_grad():
                output = self.model(input_ids=input_ids_slice)
                logits_i = output.logits[:, -1, :]  # logits for the current token
                probs = torch.softmax(logits_i, dim=-1)
                token_id = input_ids[0, i]
                token_prob = probs[0, token_id]
                # Add log probability
                total_log_prob += torch.log(token_prob + 1e-12).item()

        return total_log_prob

    def get_score(self, sequence: str) -> float:
        """
        Obtain the detector score for the sequence.

        This method can be customized depending on the detector's interface.
        For simplicity, we assume a log-likelihood score or probability-based score.
        Here, for illustration, we return the mean token probability as a proxy.
        In practice, this would interface with an external detector.

        Args:
            sequence (str): Text sequence to score.

        Returns:
            float: 'Human-ness' score, higher indicates more human-like.
        """
        # For this implementation, assume a placeholder score
        # E.g., a language model probability or detector API call
        # Here, return a dummy value; replace with actual detector API call if available
        # For example purposes, using negative perplexity as score:
        # negative_perplexity can be used; or integrate with detector API

        # Placeholder: assuming higher score means more human
        # For realistic scenario, replace with detector API call
        return 0.5  # Placeholder score

    def get_log_prob(self, sequence: str, input_prompt: str) -> float:
        """
        Compute and return the log probability of the sequence conditioned on the prompt.

        Args:
            sequence (str): The output sequence.
            input_prompt (str): The input prompt.

        Returns:
            float: Log-likelihood score.
        """
        return self.log_prob(sequence, input_prompt)
```

## requirements.txt

# requirements.txt

transformers==4.29.1  
torch==2.0.0  
numpy==1.23.5  
scipy==1.10.1  
tqdm==4.65.0  
requests==2.31.0

## trainer.py

```python
## trainer.py
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import copy
from tqdm import tqdm
from utils import preference_probability, normalize_scores, preference_label
from transformers import AdamW
from typing import List, Dict
import torch.nn.functional as F

class PPOTrainer:
    """
    Implements the PPO-based reinforcement learning fine-tuning for a language model
    using the Direct Preference Optimization (DPO) approach guided by detector scores.
    """

    def __init__(
        self,
        model,
        dataset,
        detector,
        kl_coeff=0.5,
        beta=0.5,
        lr=1e-5,
        batch_size=16,
        total_steps=30000,
        save_interval=5000,
        device='cuda',
        sequence_length=120,
        temperature=0.7,
        top_p=0.9
    ):
        """
        Initializes the PPOTrainer with model, data, detector, and hyperparameters.
        Args:
            model (ModelWrapper): Pretrained model to be fine-tuned.
            dataset (DatasetLoader): DatasetLoader providing prompt-response pairs.
            detector (DetectorAPI): Detector interface for scoring responses.
            kl_coeff (float): Coefficient for KL divergence regularization.
            beta (float): Reward scaling factor in DPO.
            lr (float): Learning rate for optimizer.
            batch_size (int): Batch size for updates.
            total_steps (int): Total training steps.
            save_interval (int): Steps interval for checkpoint saving.
            device (str): 'cuda' or 'cpu'.
            sequence_length (int): Max tokens for generation.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling cutoff.
        """
        self.model = model
        self.dataset = dataset
        self.detector = detector
        self.kl_coeff = kl_coeff
        self.beta = beta
        self.lr = lr
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.save_interval = save_interval
        self.device = device
        self.sequence_length = sequence_length
        self.temperature = temperature
        self.top_p = top_p

        # Initialize optimizer
        self.optimizer = AdamW(self.model.model.parameters(), lr=self.lr)

        # Save reference model parameters for KL divergence
        self.ref_model_params = copy.deepcopy(self.model.model.state_dict())

        # Prepare data loader
        self.data_loader = self._create_data_loader()

    def _create_data_loader(self):
        """
        Creates an infinite generator over the dataset for training.
        """
        data = self.dataset.preference_pairs
        def infinite_gen():
            while True:
                indices = np.random.permutation(len(data))
                for idx in indices:
                    yield data[idx]
        return infinite_gen()

    def _compute_kl_divergence(self):
        """
        Estimates the KL divergence between current model and reference model.
        Uses samples generated from current policy.
        """
        # Sample a batch of prompts
        batch_prompts = [pair[0] for pair in self.dataset.sample_pairs(self.batch_size)]
        responses = []
        log_probs = []

        # Generate responses and compute log probs
        for prompt in batch_prompts:
            resp = self.model.generate(
                prompt, max_new_tokens=self.sequence_length,
                temperature=self.temperature, top_p=self.top_p
            )
            responses.append((prompt, resp))
            log_prob = self.model.log_prob(resp, prompt)
            log_probs.append(log_prob)

        # Compute log probabilities under reference model (assumed to be frozen)
        ref_log_probs = []
        for (prompt, resp) in responses:
            # For simplicity assuming reference model's log probability is similar to current
            # In practice, load reference model separately or compute via stored parameters
            ref_log_prob = self.model.log_prob(resp, prompt)  # placeholder
            ref_log_probs.append(ref_log_prob)

        # Approximate KL divergence
        kl_values = []
        for log_p, log_q in zip(log_probs, ref_log_probs):
            kl = log_p - log_q  # simple difference as a proxy
            kl_values.append(kl)
        return torch.mean(torch.stack(kl_values))

    def train(self):
        """
        Main training loop for RL fine-tuning using PPO with DPO loss.
        """
        for step in tqdm(range(self.total_steps)):
            # Sample a batch of preference pairs
            batch = list(next(self.data_loader) for _ in range(self.batch_size))
            prompts = [item[0] for item in batch]
            responses_w = [item[1] for item in batch]
            responses_l = [item[2] for item in batch]
            labels = [item[3] for item in batch]  # 1 if y_w preferred, 0 if y_l preferred

            # Generate responses from current model
            # For each prompt, generate responses for both options (simulate responses)
            gen_responses_w = []
            gen_responses_l = []
            for prompt in prompts:
                resp_w = self.model.generate(
                    prompt, max_new_tokens=self.sequence_length,
                    temperature=self.temperature, top_p=self.top_p
                )
                resp_l = self.model.generate(
                    prompt, max_new_tokens=self.sequence_length,
                    temperature=self.temperature, top_p=self.top_p
                )
                gen_responses_w.append(resp_w)
                gen_responses_l.append(resp_l)

            # Compute log probabilities under current model
            log_probs_w = [self.model.log_prob(resp, prompt) for resp, prompt in zip(gen_responses_w, prompts)]
            log_probs_l = [self.model.log_prob(resp, prompt) for resp, prompt in zip(gen_responses_l, prompts)]

            # Obtain detector scores
            scores_w = [self.detector.score(resp) for resp in gen_responses_w]
            scores_l = [self.detector.score(resp) for resp in gen_responses_l]

            # Compute \(\Delta R\), preference probabilities
            delta_scores = [s_w - s_l for s_w, s_l in zip(scores_w, scores_l)]
            pref_probs = [preference_probability(s_w, s_l, scale='log_prob') for s_w, s_l in zip(scores_w, scores_l)]

            # Calculate \(\mathcal{L}_{DPO}\): negative log likelihood
            # To avoid numerical issues, add epsilon
            epsilon = 1e-8
            losses = [-np.log(p + epsilon) for p in pref_probs]
            l_dpo = torch.mean(torch.tensor(losses, dtype=torch.float32))

            # Estimate KL divergence
            kl_div = self._compute_kl_divergence()

            # Total loss
            total_loss = l_dpo + self.kl_coeff * kl_div

            # Gradient step
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Logging
            if step % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_{step}.pt')
            if step % 1000 == 0:
                print(f"Step {step}: Total Loss={total_loss.item():.4f}, "
                      f"KL={kl_div.item():.4f}, DPO Loss={l_dpo.item():.4f}")

    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.
        """
        torch.save(self.model.model.state_dict(), path)
```

## utils.py

```python
## utils.py
import numpy as np
from scipy.special import expit  # Sigmoid function
from sklearn.metrics import roc_auc_score
import torch
from typing import List, Tuple, Dict
import torch.nn.functional as F

def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize a list of scores to the [0, 1] range using min-max scaling.
    Handles negative scores such as log probabilities appropriately.
    
    Args:
        scores (List[float]): Raw detector scores or scores to normalize.
    
    Returns:
        List[float]: Normalized scores scaled between 0 and 1.
    """
    scores_arr = np.array(scores)
    min_score = np.min(scores_arr)
    max_score = np.max(scores_arr)
    if max_score - min_score < 1e-8:
        # Prevent division by zero; if all scores identical, return 0.5 for all
        return [0.5 for _ in scores]
    normalized = (scores_arr - min_score) / (max_score - min_score)
    return normalized.tolist()

def compute_preference(p1_score: float, p2_score: float, scale: str = 'log_prob') -> float:
    """
    Compute the probability that response 1 (w) is preferred over response 2 (l),
    based on their scores, using the Bradley-Terry model.
    
    Args:
        p1_score (float): Score for response w.
        p2_score (float): Score for response l.
        scale (str): Score type, default 'log_prob' indicating scores are log probabilities.
                     If 'prob', scores are assumed in [0,1].
    
    Returns:
        float: Preference probability p(y_w > y_l).
    """
    if scale == 'log_prob':
        # Assume p1_score and p2_score are in the log domain
        diff = p1_score - p2_score
        return expit(diff)  # σ(diff)
    elif scale == 'prob':
        # Scores are probabilities in [0, 1]
        diff = p1_score - p2_score
        return expit(diff)
    else:
        raise ValueError(f"Unknown scale: {scale}")

def preference_label(score_w: float, score_l: float) -> int:
    """
    Assign binary preference label: 1 if y_w is preferred over y_l, else 0.
    
    Args:
        score_w (float): Score (detector or log prob) for response w.
        score_l (float): Score for response l.
    
    Returns:
        int: 1 if w preferred, 0 otherwise.
    """
    return 1 if score_w > score_l else 0

def preference_probability(score_w: float, score_l: float) -> float:
    """
    Return the preference probability that y_w is preferred over y_l.
    Wraps compute_preference with scale='log_prob' by default.
    
    Args:
        score_w (float): Score for response w.
        score_l (float): Score for response l.
    
    Returns:
        float: Probability that y_w is preferred over y_l.
    """
    return compute_preference(score_w, score_l, scale='log_prob')

def sample_batch(dataset: List[Tuple], batch_size: int) -> List[Tuple]:
    """
    Randomly sample a batch of data tuples from the dataset.
    
    Args:
        dataset (List[Tuple]): Dataset of data points, e.g., (prompt, response, score).
        batch_size (int): Number of samples to draw.
    
    Returns:
        List[Tuple]: Randomly sampled batch.
    """
    import random
    return random.sample(dataset, min(batch_size, len(dataset)))

def generate_samples(prompt: str, model, max_tokens: int, temperature: float, top_p: float) -> str:
    """
    Generate a continuation from a model given a prompt.
    
    Args:
        prompt (str): Input prompt string.
        model: Model wrapper with generate method.
        max_tokens (int): Max number of tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
    
    Returns:
        str: Generated text continuation.
    """
    generated_text = model.generate(
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    return generated_text

def compute_auroc(labels: List[int], scores: List[float]) -> float:
    """
    Compute the AUROC score given binary labels and scores.
    
    Args:
        labels (List[int]): Ground truth labels (0 or 1 or other binary labels).
        scores (List[float]): Scores assigned by detector.
    
    Returns:
        float: AUROC score.
    """
    return roc_auc_score(labels, scores)

def compute_perplexity(texts: List[str], model, tokenizer) -> float:
    """
    Compute the average perplexity over a list of texts given a language model.
    Uses negative log likelihoods from the model head.
    
    Args:
        texts (List[str]): List of text strings.
        model: Language model for evaluation.
        tokenizer: Corresponding tokenizer.
    
    Returns:
        float: Average perplexity over the texts.
    """
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            input_ids = encodings['input_ids'].to(next(model.parameters()).device)
            attention_mask = encodings['attention_mask'].to(next(model.parameters()).device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            # Negative log likelihood
            neg_log_likelihood = outputs.loss * input_ids.shape[1]
            total_nll += neg_log_likelihood.item()
            total_tokens += input_ids.shape[1]
    if total_tokens == 0:
        return float('nan')
    avg_nll = total_nll / total_tokens
    perplexity = np.exp(avg_nll)
    return perplexity
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\llm-detector-evasion\llm-detector-evasion_repo`
