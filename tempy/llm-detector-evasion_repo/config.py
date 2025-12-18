# config.py
"""
Configuration module for reproducing "LANGUAGE MODEL DETECTORS ARE EASILY OPTIMIZED AGAINST".
It encapsulates all hyperparameters, datasets, detector settings, and model configurations
as specified in 'config.yaml'.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class DetectorConfig:
    name: str
    # Additional parameters like API endpoint or model path can be added here
    # For now, only name as identifier is used; interfacing modules handle specifics

@dataclass
class DatasetConfig:
    size_pref_pairs: int = 30000  # Number of preference pairs to generate
    prompt_dataset: str = 'OpenWebText'  # Dataset source, e.g., 'OpenWebText' or instruction datasets
    response_generation_method: str = 'model'  # 'model' (generate responses) or 'human'
    detector_score_threshold: float = 0.0  # Threshold to interpret detector scores (0 for continuous)

@dataclass
class ModelConfig:
    name: str = 'Llama-2-7b'  # Model name as per experiment
    device: str = 'cuda'      # Device to load model onto
    load_from_checkpoint: bool = False  # Whether to load from a checkpoint or train from scratch

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-5  # Optimizer learning rate
    batch_size: int = 256        # Batch size during fine-tuning
    epochs: int = 3              # Number of training epochs
    beta: float = 0.5            # KL regularization coefficient; tune as 0.05, 0.5, 5
    max_response_tokens: int = 250  # Max tokens in generated responses
    generation_temperature: float = 1.0  # Sampling temperature for response generation

@dataclass
class EvaluationConfig:
    evaluate_transferability: bool = True  # Whether to evaluate detector transferability
    evaluate_sequence_length: bool = True  # Whether to test robustness on longer outputs
    human_evaluation: bool = True          # Whether human evaluation is performed
    human_eval_samples: int = 182           # Number of samples for human evaluation
    human_eval_text_length: int = 128        # Response length in tokens for human eval

@dataclass
class DetectorSet:
    open_source: List[str] = field(default_factory=lambda: [
        'RoBERTa-large',
        'RoBERTa-base',
        'DetectGPT',
        'DetectLLM'
    ])
    commercial: List[str] = field(default_factory=lambda: [
        'GPTZero',
        'Originality.ai',
        'Winston AI'
    ])

@dataclass
class Config:
    detectors: DetectorSet = field(default_factory=DetectorSet)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

# Instantiate the config to be imported elsewhere
config = Config()
