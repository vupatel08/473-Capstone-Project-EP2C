## config.py

import os
from typing import Dict, Any
import yaml

class Config:
    """
    Centralized configuration class for data paths, model parameters,
    training, explanation, hardware, evaluation, and save paths.
    Loads settings from 'config.yaml'.
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration from YAML file
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Data paths
        self.dataset_paths: Dict[str, str] = cfg.get("dataset_paths", {
            "histopathology": "/path/to/histopathology/data",
            "toy_data": "/path/to/toy/dataset"
        })

        # Model parameters
        self.model_parameters: Dict[str, Any] = cfg.get("model_parameters", {
            "model_type": "attention",            # options: "attention", "transformer", "additive"
            "hidden_dim": 512,
            "feature_extractor": "resnet18",      # pretrained CNN backbone
            "freeze_feature_extractor": True      # freeze CNN during training
        })

        # Training settings
        self.training: Dict[str, Any] = cfg.get("training", {
            "learning_rate": 0.002,
            "batch_size": 32,
            "epochs": 1000,
            "optimizer": "Adam",
            "dropout": 0.0
        })

        # Explanation method configuration
        self.explanation_method: Dict[str, Any] = cfg.get("explanation_method", {
            "method": "xMIL-LRP",                # "xMIL-LRP", "IG", "G×I", "attention_rollout"
            "relevance_rules": {
                "linear": "LRP-epsilon",         # propagation rule for linear layers
                "attention": "AH-rule",          # propagation rule for attention modules
                "layer_norm": "LN-rule"          # propagation rule for layer norm
            }
        })

        # Hardware setup
        self.hardware: Dict[str, Any] = cfg.get("hardware", {
            "device": "cuda",                     # "cuda" or "cpu"
            "gpus": 1
        })

        # Evaluation setup
        self.evaluation: Dict[str, Any] = cfg.get("evaluation", {
            "perturbation_steps": 100,
            "metrics": {
                "AUPRC2": True,
                "AUPC": True
            },
            "visualization": {
                "heatmaps": True
            }
        })

        # Save paths
        self.save: Dict[str, str] = cfg.get("save", {
            "model_checkpoint_path": "./checkpoints/",
            "explanation_heatmaps_path": "./heatmaps/"
        })

        # Validate paths exist or create directories
        self._validate_paths()

    def _validate_paths(self):
        # Create directories if they do not exist
        for path in [self.save["model_checkpoint_path"], self.save["explanation_heatmaps_path"]]:
            if not os.path.exists(path):
                os.makedirs(path)

    def get(self) -> Dict[str, Any]:
        """
        Return the complete configuration as a nested dictionary.
        """
        return {
            "dataset_paths": self.dataset_paths,
            "model_parameters": self.model_parameters,
            "training": self.training,
            "explanation_method": self.explanation_method,
            "hardware": self.hardware,
            "evaluation": self.evaluation,
            "save": self.save
        }

# Instantiate a singleton configuration object
config = Config()

# Usage example:
# cfg = config.get()
