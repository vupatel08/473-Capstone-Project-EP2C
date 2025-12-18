## utils.py
import os
import json
import random
import yaml
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset
from torch.nn import modules as nn_modules
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoTokenizer, AutoModel, AdamW, get_scheduler

try:
    import rouge_score
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    rouge_score = None
    sentence_bleu = None
    meteor_score = None

# Load configuration from 'config.yaml'
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# ------------------ Logging Setup ------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ------------------ Seed setting for reproducibility ------------------
def set_seed(seed: int = 42) -> None:
    """
    Set random seed for all relevant libraries for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

# ------------------ Tokenization utilities ------------------
def tokenize_protein(sequence: str, tokenizer: Optional[Any] = None) -> Dict[str, Any]:
    """
    Tokenize a protein sequence using the specified tokenizer.
    Args:
        sequence (str): Amino acid sequence string.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the PLM.
    Returns:
        dict: Dictionary with input_ids and attention_mask.
    """
    if tokenizer is None:
        # Load from config
        plm_name = CONFIG['model'].get('plm_name', 'facebook/esm2_t6_8a_14B')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(plm_name)
    encoded = tokenizer(sequence, return_tensors='pt', padding='max_length', truncation=True)
    return encoded

def tokenize_text(text: str, tokenizer: Optional[Any] = None, max_length: int = 512) -> Dict[str, Any]:
    """
    Tokenize textual data with specified maximum length.
    Args:
        text (str): Input text string.
        tokenizer: Tokenizer compatible with the text encoder.
        max_length (int): Max token length.
    Returns:
        dict: Contains input_ids and attention_mask tensors.
    """
    if tokenizer is None:
        # Load from config
        text_enc_name = CONFIG['model'].get('text_encoder_name', 'allenai/scibert_scivocab_uncased')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(text_enc_name)
    encoded = tokenizer(
        text, 
        max_length=max_length, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )
    return encoded

# ------------------ Checkpoint management ------------------
def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    step: int, 
    path: str
) -> None:
    """
    Save model and optimizer state dicts at given step.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved at step {step} to {path}")

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str
) -> int:
    """
    Load checkpoint weights into model and optimizer.
    Returns:
        int: Last trained step.
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_step = checkpoint.get('step', 0)
    logger.info(f"Loaded checkpoint from {path} at step {last_step}")
    return last_step

# ------------------ Learning rate scheduler ------------------
def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 10000,
    schedule_type: str = 'linear_warmup'
) -> _LRScheduler:
    """
    Return scheduler: linear warmup + cosine decay or other as needed.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            import math
            return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

# ------------------ Gradient clipping ------------------
def clip_gradients(model: torch.nn.Module, clip_value: float = 1.0) -> None:
    """
    Clip gradients to prevent exploding gradients.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
    logger.debug("Gradients clipped to max norm {}".format(clip_value))

# ------------------ Logging metrics ------------------
def log_metrics(
    metrics: Dict[str, float], 
    step: int, 
    log_dir: str = './logs'
) -> None:
    """
    Log metrics as JSON line or to a dashboard.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'metrics.jsonl')
    record = {'step': step}
    record.update(metrics)
    with open(log_path, 'a') as f:
        f.write(json.dumps(record) + '\n')
    logger.info(f"Step {step}: {metrics}")

# ------------------ External API call wrapper (GPT) ------------------
import time
import requests

def call_openai_api(
    prompt: str,
    model: str = 'gpt-4',
    max_tokens: int = 512,
    api_key: str = '',  # User must set their API key
    temperature: float = 0.7,
    retries: int = 3,
    backoff_factor: float = 2.0
) -> str:
    """
    Call OpenAI API with retries and backoff.
    """
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': model,
        'prompt': prompt,
        'max_tokens': max_tokens,
        'temperature': temperature
    }
    for attempt in range(retries):
        try:
            response = requests.post('https://api.openai.com/v1/completions', headers=headers, json=payload)
            response.raise_for_status()
            reply_json = response.json()
            # Assume first choice
            reply_text = reply_json['choices'][0]['text'].strip()
            return reply_text
        except requests.RequestException as e:
            logger.warning(f"OpenAI API call failed ({e}), attempt {attempt + 1}")
            time.sleep(backoff_factor ** attempt)
    logger.error("OpenAI API call failed after retries")
    return ""

# ------------------ Save/load dataset helper functions ------------------
def load_dataset(path: str) -> Dataset:
    """
    Load dataset from given path, assuming JSON or CSV format.
    """
    # Placeholder: Implement as needed per dataset format
    # For now, return an empty list or dummy dataset
    class DummyDataset(Dataset):
        def __len__(self):
            return 0
        def __getitem__(self, idx):
            return {}
    return DummyDataset()

# ------------------ Set up environment and configuration validation ------------------
def validate_config():
    """
    Validate essential configuration entries.
    """
    required_keys = ['training', 'model', 'dataset', 'prompt_templates', 'evaluation', 'hardware']
    for key in required_keys:
        if key not in CONFIG:
            logger.warning(f"Configuration missing key: {key}")
    # Further validation can be added here
