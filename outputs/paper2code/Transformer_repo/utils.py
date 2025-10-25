"""utils.py
This module holds common helper functions shared by Trainer, Evaluation, and other modules.
It provides:
  - Configuration parsing from a YAML file.
  - A learning rate scheduler function according to the Transformer paper.
  - A checkpoint averaging utility that averages multiple PyTorch model state dictionaries.
  - Logging setup helpers.
  - Additional utility functions for debugging and file existence checks.
  
Dependencies:
  - PyYAML (for YAML parsing)
  - torch (for checkpoint operations)
  - logging, os, math, json (standard libraries)
  
All functions use strong type annotations and default values where needed.
"""

import os
import math
import logging
import json
from typing import Dict, Any, List, Optional

import torch
import yaml


def parse_config(config_path: str) -> Dict[str, Any]:
    """
    Parses a YAML configuration file and returns a configuration dictionary.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        config (Dict[str, Any]): Dictionary containing configuration parameters.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If required sections are missing in the configuration.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at path: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config: Dict[str, Any] = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {exc}")
    
    # Validate that required sections are present.
    required_sections: List[str] = ["training", "hyperparameters", "dataset", "inference"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required configuration section '{section}' is missing.")
    
    return config


def get_learning_rate(d_model: int, step: int, warmup_steps: int) -> float:
    """
    Computes the learning rate at a given training step as per the Transformer paper.
    
    The formula is:
      lr = (d_model)^(-0.5) * min(step^(-0.5), step * (warmup_steps)^(-1.5))
    
    Args:
        d_model (int): The dimension of the model embeddings.
        step (int): The current training step (should be >= 1).
        warmup_steps (int): The number of warmup steps.
    
    Returns:
        lr (float): The computed learning rate.
    
    Note:
        If step is less than 1, it defaults to 1 to avoid division by zero.
    """
    current_step: int = step if step >= 1 else 1
    lr: float = (d_model ** -0.5) * min(current_step ** -0.5, current_step * (warmup_steps ** -1.5))
    return lr


def average_checkpoints(checkpoint_paths: List[str]) -> Dict[str, torch.Tensor]:
    """
    Averages the model state dictionaries from a list of checkpoint file paths.
    
    For every key in the state dictionaries, an element-wise average of the tensors
    is computed and returned as a new state dictionary.
    
    Args:
        checkpoint_paths (List[str]): List of file paths to PyTorch checkpoint state dicts.
    
    Returns:
        averaged_state (Dict[str, torch.Tensor]): A state dictionary with averaged parameters.
    
    Raises:
        ValueError: If no checkpoint paths are provided, or if checkpoint keys/shapes do not match.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided for averaging.")
    
    averaged_state: Dict[str, torch.Tensor] = {}
    num_checkpoints: int = len(checkpoint_paths)
    
    for idx, path in enumerate(checkpoint_paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        checkpoint_state: Dict[str, torch.Tensor] = torch.load(path, map_location="cpu")
        
        if idx == 0:
            # Initialize the averaged_state dictionary with clones of the first checkpoint's values.
            for key, value in checkpoint_state.items():
                averaged_state[key] = value.clone().float()
        else:
            # Ensure that keys match between checkpoints.
            if checkpoint_state.keys() != averaged_state.keys():
                raise ValueError(f"Checkpoint keys do not match in file: {path}")
            for key, value in checkpoint_state.items():
                if averaged_state[key].shape != value.shape:
                    raise ValueError(f"Shape mismatch for key '{key}' in checkpoint: {path}")
                averaged_state[key] += value.clone().float()
    
    # Divide each parameter by the number of checkpoints.
    for key in averaged_state.keys():
        averaged_state[key] /= num_checkpoints
    
    return averaged_state


def setup_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up the logger to output messages to the console and optionally to a file.
    
    Args:
        log_file (Optional[str]): Path to a log file. If provided, logs are written to the file as well.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG). Default is logging.INFO.
    
    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    logger: logging.Logger = logging.getLogger("TransformerLogger")
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicate logs.
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter: logging.Formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler.
    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified.
    if log_file:
        file_handler: logging.FileHandler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def pretty_print_config(config: Dict[str, Any]) -> None:
    """
    Pretty prints the configuration dictionary in JSON format.
    
    Args:
        config (Dict[str, Any]): The configuration dictionary.
    """
    pretty_config: str = json.dumps(config, indent=2)
    print(pretty_config)


def check_file_exists(file_path: str) -> bool:
    """
    Checks whether a file exists at the given path.
    
    Args:
        file_path (str): The path to the file.
    
    Returns:
        exists (bool): True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)


# Example usage of the utility functions (for testing purposes).
if __name__ == "__main__":
    # Set up logger for testing.
    test_logger: logging.Logger = setup_logger(level=logging.DEBUG)
    test_logger.info("Testing utils.py functionalities...")
    
    # Test configuration parsing.
    test_config_path: str = "config.yaml"
    try:
        config: Dict[str, Any] = parse_config(test_config_path)
        test_logger.info("Configuration parsed successfully:")
        pretty_print_config(config)
    except Exception as e:
        test_logger.error(f"Failed to parse configuration: {e}")
    
    # Test learning rate scheduler.
    d_model_test: int = config.get("hyperparameters", {}).get("base_model", {}).get("d_model", 512)
    warmup_steps_test: int = config.get("training", {}).get("warmup_steps", 4000)
    for step in [0, 1, 1000, 10000]:
        lr: float = get_learning_rate(d_model_test, step, warmup_steps_test)
        test_logger.info(f"Learning rate at step {step}: {lr:.6f}")
    
    # Test checkpoint averaging (assuming dummy checkpoints exist).
    # For demonstration, create two dummy state dicts and save them temporarily.
    dummy_state1: Dict[str, torch.Tensor] = {"param1": torch.ones(2, 2), "param2": torch.ones(3)}
    dummy_state2: Dict[str, torch.Tensor] = {"param1": 2 * torch.ones(2, 2), "param2": 2 * torch.ones(3)}
    dummy_path1: str = "dummy_checkpoint1.pt"
    dummy_path2: str = "dummy_checkpoint2.pt"
    torch.save(dummy_state1, dummy_path1)
    torch.save(dummy_state2, dummy_path2)
    
    try:
        averaged_state: Dict[str, torch.Tensor] = average_checkpoints([dummy_path1, dummy_path2])
        test_logger.info("Averaged checkpoint state:")
        for key, tensor in averaged_state.items():
            test_logger.info(f"{key}: {tensor}")
    except Exception as error:
        test_logger.error(f"Checkpoint averaging failed: {error}")
    finally:
        # Clean up dummy checkpoint files.
        if os.path.exists(dummy_path1):
            os.remove(dummy_path1)
        if os.path.exists(dummy_path2):
            os.remove(dummy_path2)
    
    test_logger.info("All utils tests completed successfully.")
