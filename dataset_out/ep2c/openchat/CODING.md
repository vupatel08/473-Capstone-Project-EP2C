# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
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
```

## evaluation.py

```python
## evaluation.py
import torch
from typing import List, Dict, Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
import os
import re
import json

from model import Model


class Evaluation:
    """
    Handles model inference, response scoring, and benchmark evaluation.
    Uses external LLM (e.g., GPT-4) for response scoring based on a prompt template.
    """

    def __init__(
        self,
        model: Model,
        eval_dataset: List[Dict],
        benchmark_name: str = "CustomEval",
        scoring_model_name: str = "gpt-4",
        num_eval_samples: int = 128,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        batch_size: int = 8,
        eval_benchmarks: Optional[List[str]] = None
    ):
        """
        Initializes the evaluation with model and dataset.
        """
        self.model = model
        self.eval_dataset = eval_dataset
        self.num_eval_samples = num_eval_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.scoring_model_name = scoring_model_name
        self.benchmark_name = benchmark_name
        self.eval_benchmarks = eval_benchmarks or ["CustomEval"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the scoring language model
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(scoring_model_name)
        self.scoring_model = AutoModelForCausalLM.from_pretrained(scoring_model_name).to(self.device)
        self.scoring_model.eval()

        # Load any benchmark-specific data or set prompts
        # For simplicity, assume eval_dataset is list of dicts with 'prompt' and optional other info
        
        # Compile a regex for extracting evaluation scores from GPT responses, if needed
        self.score_regex = re.compile(r"Rating:\s*\[(\d+\.?\d*)\]", re.IGNORECASE)

    def generate_response(self, prompt: str, class_label: Optional[str] = None) -> str:
        """
        Generate a model response for a given prompt with optional class conditioning.
        """
        # Use model's generate method
        response = self.model.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
        )
        return response.strip()

    def score_response(self, prompt: str, response: str) -> float:
        """
        Scores the response using an external language model (e.g., GPT-4), via a prompt template.
        Returns a score between 1 and 10.
        """
        eval_prompt = (
            "[Instruction] Please act as an impartial judge and evaluate the quality of the response "
            "provided by an AI assistant to the conversation displayed below. Your evaluation should "
            "consider factors such as helpfulness, relevance, accuracy, depth, creativity, and "
            "level of detail of the response. Begin your evaluation by providing a short explanation. "
            "Be as objective as possible. After providing your explanation, you must rate the response "
            "on a scale of 1 to 10 by strictly following this format:\n"
            "\"Rating: [[number]]\".\n"
            "[Conversation]\n"
            f"{prompt}\n"
            "[The Start of Assistant's Response]\n"
            f"{response}"
        )

        # Call external language model API (e.g., OpenAI GPT) - placeholder for actual API call
        # Here, for demonstration, assume a function 'call_llm' exists
        llm_response = self._call_llm(eval_prompt)

        # Parse score from response
        match = self.score_regex.search(llm_response)
        if match:
            score_str = match.group(1)
            try:
                score = float(score_str)
                score = max(1.0, min(10.0, score))
                return score
            except ValueError:
                pass
        # Fallback: parse numerical value manually
        try:
            # Extract first number in response
            num_match = re.search(r"\d+\.?\d*", llm_response)
            if num_match:
                score = float(num_match.group(0))
                score = max(1.0, min(10.0, score))
                return score
        except:
            pass
        # If parsing fails, fallback to default score
        return 5.0

    def _call_llm(self, prompt: str) -> str:
        """
        Placeholder for calling the external LLM API such as OpenAI GPT.
        Implement your API call here. For example, using OpenAI API.
        """
        # import openai
        # response = openai.ChatCompletion.create(
        #     model=self.scoring_model_name,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.0,
        #     max_tokens=100,
        #     n=1,
        # )
        # return response.choices[0].message['content']
        # For now, just simulate response (for testing or local scripts)
        # Raise error or return dummy value
        raise NotImplementedError("Implement API call to language model for scoring.")

    def evaluate_benchmark(self, benchmark_name: str, dataset: List[Dict]) -> Dict:
        """
        Evaluate model on a specific benchmark dataset.
        Supports multiple samples per prompt and pairwise responses if provided.
        Returns metrics like win rate, average score, accuracy.
        """
        results = {
            "total": 0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "scores": [],
            "accuracy": None,  # optional
        }

        # For simplicity, assume dataset is list of dicts with 'prompt' and expected 'reference' or 'label'
        # The evaluation will generate a response per prompt and score it

        for item in tqdm.tqdm(dataset, desc=f"Eval {benchmark_name}"):
            prompt = item.get("prompt", "")
            # Generate multiple responses if needed
            responses = []
            for _ in range(self.num_eval_samples):
                try:
                    resp = self.generate_response(prompt)
                    responses.append(resp)
                except Exception as e:
                    responses.append("")
            # Score responses
            scores = []
            for resp in responses:
                try:
                    score = self.score_response(prompt, resp)
                except Exception:
                    score = 5.0
                scores.append(score)

            # Compute average response score
            avg_score = sum(scores) / len(scores)
            results["scores"].append(avg_score)

            # For pairwise comparison or baseline comparison, implement logic here
            # For simplicity, assume we compare to a baseline (e.g., previous model score)
            # or compare two responses; skipping for generality.

            # Placeholder for win/loss counting
            # In actual, compare responses (e.g., response from new model vs baseline), here assume always 'win'
            # results["wins"] += 1
            # or implement specific comparison based on reference or other metric.
            # For demonstration, assume each response is better if avg_score > 6
            if avg_score > 6:
                results["wins"] += 1
            elif avg_score == 6:
                results["ties"] += 1
            else:
                results["losses"] += 1
            results["total"] += 1

        # Aggregate metrics
        win_rate = results["wins"] / results["total"] if results["total"] > 0 else 0.0
        average_score = sum(results["scores"]) / len(results["scores"]) if results["scores"] else 0.0
        results.update(
            {
                "win_rate": win_rate,
                "average_score": average_score,
            }
        )
        return results

    def evaluate_all(self, benchmarks: List[str], datasets_map: Dict[str, List[Dict]]) -> Dict:
        """
        Run evaluation over all specified benchmarks and return aggregated metrics.
        """
        all_results = {}
        for bm in benchmarks:
            dataset = datasets_map.get(bm, [])
            res = self.evaluate_benchmark(bm, dataset)
            all_results[bm] = res
        return all_results

    def save_results(self, filepath: str):
        """
        Save evaluation results to a JSON file.
        """
        # Save current evaluation metrics
        # Can be called after evaluation_all()
        pass  # implementation as needed
```

## main.py

```python
# main.py

import os
import yaml
import torch
import random
import logging

from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Load configuration from 'config.yaml'
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Set random seed for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 3. Load dataset
    dataset_cfg = config.get('dataset', {})
    dataset_path = dataset_cfg.get('dataset_path', 'data/sharegpt_mixed_quality.json')
    train_sample_size = dataset_cfg.get('train_sample_size', 128)
    eval_sample_size = dataset_cfg.get('eval_sample_size', 128)
    dataset_loader = DatasetLoader({
        'dataset_path': dataset_path,
        'train_sample_size': train_sample_size,
        'eval_sample_size': eval_sample_size,
        'seed': seed
    })
    train_data, eval_data = dataset_loader.load_data()
    logging.info(f"Loaded dataset with {len(train_data)} training samples and {len(eval_data)} evaluation samples.")

    # 4. Initialize the language model
    model_cfg = config.get('model', {})
    pretrained_name = model_cfg.get('pretrained_model_name', "huggingface/llama-13b")
    conditioning_token = model_cfg.get('conditioning_token', "<|class|>")
    model = Model(pretrained_name, conditioning_token)
    logging.info(f"Loaded pretrained model {pretrained_name} with class conditioning token '{conditioning_token}'.")

    # 5. Initialize the trainer
    training_cfg = config.get('training', {})
    beta = training_cfg.get('beta', 0.2)
    lr = training_cfg.get('learning_rate', 3e-5)
    batch_size = training_cfg.get('batch_size', 200)
    epochs = training_cfg.get('epochs', 3)
    max_grad_norm = training_cfg.get('max_grad_norm', 1.0)
    warmup_steps = training_cfg.get('warmup_steps', 1000)
    weight_decay = training_cfg.get('weight_decay', 0.01)

    trainer = Trainer(
        model=model,
        dataset=train_data,
        beta=beta,
        learning_rate=lr,
        batch_size=batch_size,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        seed=seed
    )

    # 6. Run training
    logging.info("Starting training process...")
    trainer.train()
    output_dir = config.get('output_dir', 'output/openchat_finetuned')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    trainer.save_model(output_dir)
    logging.info(f"Model saved to {output_dir}")

    # 7. Initialize evaluation
    eval_dataset = eval_data  # For simplicity, using eval_data; could load benchmark datasets separately
    eval_instance = Evaluation(
        model=model,
        eval_dataset=eval_dataset,
        scoring_model_name='gpt-4',  # or another LLM for scoring
        num_eval_samples=128,
        max_new_tokens=256,
        temperature=0.7
    )

    # 8. Evaluate on each benchmark
    benchmarks = config.get('evaluation', {}).get('eval_benchmarks', [
        "AlpacaEval", "MT-bench", "Vicuna-bench", "AGIEval"
    ])
    logging.info(f"Starting evaluation on benchmarks: {benchmarks}")
    results = eval_instance.evaluate_all(benchmarks, datasets_map={bm: eval_dataset for bm in benchmarks})

    # 9. Log evaluation metrics
    for bm in benchmarks:
        res = results[bm]
        logging.info(f"Benchmark {bm}: Win Rate={res.get('win_rate', 0):.2%}, "
                     f"Average Score={res.get('average_score', 0):.2f}")
    # Optional: Save evaluation results to file
    eval_results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(eval_results_path, 'w') as f:
        import json
        json.dump(results, f, indent=2)
    logging.info(f"Evaluation results saved to {eval_results_path}")

# Run main when executing the script
if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    """
    Encapsulates a pre-trained LLaMA-based language model with class-conditioning support.
    Provides methods for forward inference (training) and response generation.
    """

    def __init__(self, pretrained_model_name: str = "huggingface/llama-13b", conditioning_token: str = "<|class|>"):
        """
        Load the pre-trained model and tokenizer, set up special tokens and device.

        Args:
            pretrained_model_name (str): Hugging Face model identifier.
            conditioning_token (str): Special token used for class conditioning.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conditioning_token = conditioning_token

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name).to(self.device)

        # Ensure conditioning token is in tokenizer
        if self.conditioning_token not in self.tokenizer.get_vocab():
            # Add special conditioning token
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.conditioning_token]})
            # Resize model embeddings accordingly
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.conditioning_token_id = self.tokenizer.convert_tokens_to_ids(self.conditioning_token)

        # For convenience, set to evaluation mode
        self.model.eval()

        # Optional: store a cache of class labels to prompt prefixes
        self.class_prefixes = {
            "expert": f"{self.conditioning_token} GPT4 User:",
            "sub_optimal": f"{self.conditioning_token} GPT3 User:",
            # Add more if needed
        }

    def set_conditioning(self, class_label: str) -> str:
        """
        Retrieve or create the prompt prefix for a given class label.

        Args:
            class_label (str): e.g., "expert" or "sub_optimal"

        Returns:
            prompt_prefix (str): conditioned prefix
        """
        prefix = self.class_prefixes.get(class_label, f"{self.conditioning_token} User:")
        return prefix

    def prepare_prompt(self, prompt: str, class_label: Optional[str] = None) -> torch.Tensor:
        """
        Construct the conditioned input prompt sequence.

        Args:
            prompt (str): user instruction or dialogue turn
            class_label (str, optional): class label for conditioning

        Returns:
            input_ids (torch.Tensor): tokenized input sequence
        """
        if class_label:
            prefix = self.set_conditioning(class_label)
            full_prompt = f"{prefix}\n{prompt}"
        else:
            # If no class label provided, use a default prefix
            full_prompt = prompt

        # Encode the prompt
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        return input_ids

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Run the model forward pass for training. Returns logits.

        Args:
            input_ids (torch.Tensor): input sequence
            attention_mask (torch.Tensor, optional): attention mask

        Returns:
            logits (torch.Tensor): output logits, shape (batch_size, seq_len, vocab_size)
        """
        # Model expects batch dimension; assume batch size=1 here
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits  # shape: (1, seq_len, vocab_size)
        return logits

    def generate(
        self,
        prompt: str,
        class_label: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """
        Generate a response conditioned on the prompt and optional class label.

        Args:
            prompt (str): user input prompt
            class_label (str, optional): conditioning class label
            max_new_tokens (int): maximum tokens to generate
            temperature (float): sampling temperature
            do_sample (bool): whether to sample or greedy decode
            top_p (float): nucleus sampling probability
            top_k (int): top-k sampling

        Returns:
            response (str): generated text response
        """
        # Prepare conditioned prompt
        conditioned_prompt = self.prepare_prompt(prompt, class_label)

        # Generate response
        output_ids = self.model.generate(
            input_ids=conditioned_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=1,
        )

        # Decode output
        decoded_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract the reply part: remove the prompt leading part
        # Here, we naive strip the prompt part out, assuming generate appends to prompt
        response = decoded_text[len(self.tokenizer.decode(conditioned_prompt[0], skip_special_tokens=True)) :].strip()

        return response

    def save(self, save_path: str):
        """
        Save model and tokenizer to disk.

        Args:
            save_path (str): directory path to save the model
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load(self, load_path: str):
        """
        Load model and tokenizer from disk.

        Args:
            load_path (str): directory path where model is saved
        """
        self.model = AutoModelForCausalLM.from_pretrained(load_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
```

## trainer.py

```python
## trainer.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from typing import List, Dict, Tuple, Optional
import math
import tqdm

from dataset_loader import DataEntry
from model import Model

class Trainer:
    """
    Fine-tunes a class-conditioned language model using reward-weighted supervised learning
    based on the C-RLFT approach described in the paper.
    """
    def __init__(
        self,
        model: Model,
        dataset: List[Dict],
        beta: float = 0.2,
        learning_rate: float = 3e-5,
        batch_size: int = 200,
        epochs: int = 3,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01,
        eval_interval: int = 1,
        max_steps: Optional[int] = None,
        output_dir: str = "output/openchat_finetuned",
        seed: int = 42
    ):
        """
        Initializes the trainer with model, dataset, hyperparameters.
        """
        self.model = model
        self.dataset = dataset
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.eval_interval = eval_interval
        self.max_steps = max_steps
        self.output_dir = output_dir
        self.seed = seed

        # Prepare device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set seed for reproducibility
        torch.manual_seed(self.seed)

        # Prepare the DataLoader
        self._prepare_dataloader()

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Prepare learning rate scheduler
        total_training_steps = self.max_steps if self.max_steps else len(self.dataloader) * self.epochs
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_training_steps,
        )

    def _prepare_dataloader(self):
        """
        Converts dataset list into DataLoader with batching.
        """
        self.train_dataset = self.dataset
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Collate batch: process list of dicts into batched tensors.
        """
        prompts = [item['prompt'] for item in batch]
        responses = [item['response'] for item in batch]
        class_labels = [item['class_label'] for item in batch]
        rewards = [item['reward'] for item in batch]

        # Tokenize prompts and responses together, with class conditioning
        # For simplicity, concatenate prompt + response, separated by special token if needed
        # but here, assume prompt and response are concatenated directly
        # Compatibility with model.py's prepare_prompt method

        # Using model's tokenizer
        tokenized_prompts = [self.model.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False) for prompt in prompts]
        tokenized_responses = [self.model.tokenizer.encode(response, return_tensors='pt', add_special_tokens=False) for response in responses]

        # Concatenate input_ids: prompt + response with separator if necessary
        input_ids_list = []
        attention_mask_list = []

        for p_ids, r_ids in zip(tokenized_prompts, tokenized_responses):
            input_ids = torch.cat([p_ids, r_ids], dim=1).squeeze(0)
            input_ids_list.append(input_ids)
            attention_mask_list.append(torch.ones_like(input_ids))

        # Pad sequences to the same length
        max_len = max([ids.shape[0] for ids in input_ids_list])
        input_ids_padded = []
        attention_mask_padded = []

        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - ids.shape[0]
            input_ids_padded.append( torch.cat([ids, torch.full((pad_len,), self.model.tokenizer.pad_token_id, dtype=torch.long)]) )
            attention_mask_padded.append( torch.cat([mask, torch.zeros(pad_len, dtype=torch.float)]) )

        batch_input_ids = torch.stack(input_ids_padded).to(self.device)
        batch_attention_mask = torch.stack(attention_mask_padded).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)

        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'rewards': rewards,
            'class_labels': class_labels,
            'prompts': prompts,
            'responses': responses
        }

    def train(self):
        """
        Runs the training loop over the dataset for specified epochs/steps.
        """
        global_step = 0
        total_steps = self.max_steps if self.max_steps else len(self.dataloader) * self.epochs

        for epoch in range(1, self.epochs + 1):
            epoch_iterator = tqdm.tqdm(self.dataloader, desc=f"Epoch {epoch}")
            for step, batch in enumerate(epoch_iterator):
                if self.max_steps and global_step >= self.max_steps:
                    break

                # Prepare data
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                responses = batch['responses']
                rewards = batch['rewards']
                class_labels = batch['class_labels']
                prompts = batch['prompts']

                # Compute reward weights w_i
                # r_c: 1.0 for 'expert', alpha=0.8 for 'sub_optimal'
                # w_i = exp( (1/beta) * r_c )
                reward_values = torch.zeros(len(rewards), device=self.device)
                for i, c_label in enumerate(class_labels):
                    if c_label.lower() == 'expert':
                        r_c = 1.0
                    else:
                        r_c = 0.8  # same as alpha parameter
                    reward_values[i] = torch.exp( (1.0 / self.beta) * r_c )

                # Zero gradients
                self.model.model.zero_grad()

                # Forward pass
                logits = self.model.forward(input_ids, attention_mask)

                # Compute cross-entropy loss per token
                # logits shape: (batch_size, seq_len, vocab_size)
                # goals: responses are tokenized; losses per token
                # We need to get target tokens: response part (after prompt)
                # For simplicity, assume responses are entire outputs after prompt in input_ids
                # So, compute loss over entire sequence
                # Prepare target response tokens
                # For simplicity, compute cross-entropy over entire sequence, ignoring prompt tokens:
                # Extract response tokens (assuming concatenated prompt + response)
                # Alternatively, recompute as token-based seq loss over the response part
                # Here, as we have input_ids: the entire sequence
                # We'll compute loss over entire sequence, masking prompt tokens

                seq_len = input_ids.shape[1]
                # Estimate prompt length
                prompt_lengths = [len(self.model.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
                max_prompt_len = max(prompt_lengths)
                # Create a mask for response tokens
                loss_mask = torch.zeros_like(input_ids, dtype=torch.float, device=self.device)
                for i, plen in enumerate(prompt_lengths):
                    # response tokens: positions after prompt
                    start_idx = plen
                    end_idx = input_ids.shape[1]  # end of sequence
                    loss_mask[i, start_idx:end_idx] = 1.0

                # Shift labels: next token prediction
                labels = input_ids.clone()
                labels[loss_mask == 0] = -100  # ignore prompt tokens in loss

                # Compute per token loss
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                shift_mask = loss_mask[:, 1:].contiguous()

                # Compute loss per token
                loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss_per_token = loss_per_token.view(shift_labels.shape).sum(dim=1)  # per sequence
                # Apply mask
                token_losses = []
                for i in range(shift_logits.shape[0]):
                    mask_seq = shift_mask[i]
                    token_loss_seq = loss_per_token[i] * mask_seq
                    # Average over non-zero mask tokens
                    denom = mask_seq.sum() + 1e-8
                    token_losses.append(token_loss_seq.sum() / denom)
                batch_loss = torch.stack(token_losses)  # shape: (batch_size,)

                # Weight the loss by reward weights
                weighted_loss = torch.mean(batch_loss * reward_values)

                # Backpropagation
                weighted_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), self.max_grad_norm)

                # Optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()

                global_step +=1

                # Logging
                if step % 10 == 0:
                    tqdm.tqdm.write(f"Epoch {epoch} Step {step} Loss {weighted_loss.item():.4f}")

            # Optional: Save checkpoint per epoch
            # self.save_model(f"{self.output_dir}/checkpoint_epoch_{epoch}")

    def save_model(self, path: str):
        """
        Save the trained model to path.
        """
        self.model.save(path)
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\openchat\openchat_repo`
