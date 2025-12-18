## utils.py
import os
import sys
import yaml
import json
import random
import numpy as np
import torch
import argparse
from typing import Any, Dict, Tuple, List
import logging
from functools import wraps

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        dict: Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments to override config parameters.
    Returns:
        Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Experiment configuration overrides.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--model_name", type=str, default=None, help="Model name/path override.")
    parser.add_argument("--dataset_prompt_pairs", type=str, default=None, help="Path to prompt pairs dataset.")
    parser.add_argument("--dataset_wordnet", type=str, default=None, help="Path to WordNet relations dataset.")
    parser.add_argument("--dataset_multimodal", type=str, default=None, help="Path to multimodal dataset.")
    parser.add_argument("--device", type=str, default=None, help="Device to run models on ('cuda' or 'cpu').")
    # Add more overrides as needed
    return parser.parse_args()

def load_and_override_config() -> Dict[str, Any]:
    """
    Load config.yaml and override with CLI arguments if provided.
    Returns:
        dict: Final combined configuration.
    """
    config = load_config()
    args = parse_args()
    # Override top-level keys if arguments are provided
    if args.seed is not None:
        config['sampling']['seed'] = args.seed
    if args.model_name:
        config['model']['name'] = args.model_name
    if args.dataset_prompt_pairs:
        config['dataset']['prompt_pairs_path'] = args.dataset_prompt_pairs
    if args.dataset_wordnet:
        config['dataset']['wordnet_relations_path'] = args.dataset_wordnet
    if args.dataset_multimodal:
        config['dataset']['multimodal_data_path'] = args.dataset_multimodal
    if args.device:
        config['misc']['model_device'] = args.device
    return config

def set_seed(seed: int):
    """
    Set seed for reproducibility across torch, numpy, and random.
    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_hyperparameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract hyperparameters from config with defaults.
    Args:
        config (dict): Configuration dictionary.
    Returns:
        dict: Hyperparameter dict with proper types.
    """
    hp = {}
    # Sampling hyperparameters
    hp['num_trajectories'] = int(config['sampling'].get('num_trajectories', 20))
    hp['max_length'] = int(config['sampling'].get('max_length', 20))
    hp['temperature'] = float(config['sampling'].get('temperature', 1.0))
    hp['seed'] = int(config['sampling'].get('seed', 42))
    # Divergence settings
    hp['likelihood_normalization_tau'] = float(config['divergence'].get('likelihood_normalization_tau', 0.5))
    hp['divergence_type'] = str(config['divergence'].get('type', 'log_l1'))
    # Evaluation
    hp['batch_size'] = int(config['evaluation'].get('batch_size', 32))
    # Model
    hp['model_name'] = str(config['model'].get('name', 'gpt2-large'))
    hp['model_type'] = str(config['model'].get('type', 'transformers'))
    hp['device'] = str(config['misc'].get('model_device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    # Dataset paths
    hp['prompt_pairs_path'] = str(config['dataset'].get('prompt_pairs_path', 'data/prompt_pairs.json'))
    hp['wordnet_relations_path'] = str(config['dataset'].get('wordnet_relations_path', 'data/wordnet_relations.json'))
    hp['multimodal_data_path'] = str(config['dataset'].get('multimodal_data_path', 'data/multimodal_inputs.json'))
    return hp

def load_dataset(dataset_path: str, dataset_type: str = 'prompt_pairs') -> Any:
    """
    Load dataset based on type.
    Args:
        dataset_path (str): Path to dataset file.
        dataset_type (str): 'prompt_pairs', 'wordnet_relations', 'multimodal'.
    Returns:
        Data in appropriate structure.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} not found.")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if dataset_type == 'prompt_pairs':
        # Expect list of dicts with prompt1, prompt2, label
        return data
    elif dataset_type == 'wordnet_relations':
        # List of dicts with word1, word2, relation
        return data
    elif dataset_type == 'multimodal':
        # List of dicts with image, caption, prompt
        return data
    else:
        return data

def ensure_full_stop(prompt: str) -> str:
    """
    Append full stop if missing.
    Args:
        prompt (str): Input prompt string.
    Returns:
        str: Prompt ending with '.'
    """
    prompt = prompt.strip()
    if not prompt.endswith(('.', '!', '?', ':', ';')):
        prompt += '.'
    return prompt

def prepare_prompt(prompt: str, add_full_stop: bool = True) -> str:
    """
    Prepare prompt string: optionally add full stop.
    Args:
        prompt (str): Raw prompt.
        add_full_stop (bool): Whether to append '.' if missing.
    Returns:
        str: Formatted prompt.
    """
    if add_full_stop:
        prompt = ensure_full_stop(prompt)
    return prompt

def save_json(data: Any, filepath: str):
    """
    Save data to JSON file.
    Args:
        data: Data to save.
        filepath (str): Target file path.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(filepath: str) -> Any:
    """
    Load JSON data from file.
    Args:
        filepath (str): Path to JSON file.
    Returns:
        data: Parsed JSON data.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def pretty_print(text: str):
    """
    Print formatted text with separators for clarity.
    """
    print(f"\n{'='*40}\n{text}\n{'='*40}")

def get_device(device_str: str):
    """
    Return the torch device object based on string.
    """
    if device_str.lower() == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def batchify(data_list: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split list into batches.
    Args:
        data_list: List of data items.
        batch_size: Desired batch size.
    Returns:
        List of batches (each a list).
    """
    return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]

def add_prefix_if_missing(prompt: str, prefix: str) -> str:
    """
    Add specified prefix if prompt does not start with it.
    """
    if not prompt.startswith(prefix):
        return prefix + ' ' + prompt
    return prompt
