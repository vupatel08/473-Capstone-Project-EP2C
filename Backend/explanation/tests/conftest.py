"""
Pytest fixtures and shared test data for explanation layer tests.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, List


@pytest.fixture
def sample_paper_json() -> Dict:
    """Sample paper JSON structure for testing."""
    return {
        "title": "Test Paper: Neural Architecture Search",
        "authors": ["Author One", "Author Two"],
        "url": "https://arxiv.org/abs/1234.5678",
        "abstract": "This paper presents a novel approach to neural architecture search using reinforcement learning.",
        "body_text": [
            {
                "section": "Introduction",
                "text": "Neural architecture search has become an important area of research in deep learning."
            },
            {
                "section": "Methodology",
                "text": "Our method uses a controller network to generate architectures. The controller is trained using policy gradient methods."
            },
            {
                "section": "Experiments",
                "text": "We evaluate our approach on CIFAR-10 and ImageNet datasets. We use a learning rate of 0.001 and train for 100 epochs."
            }
        ]
    }


@pytest.fixture
def sample_generated_files() -> Dict[str, str]:
    """Sample generated code files for testing."""
    return {
        "model.py": '''"""
Model implementation for the paper.
Implements Section 3.2 of the paper.
"""
class Controller:
    """Controller network for architecture search."""
    def __init__(self, hidden_size: int = 128):
        self.hidden_size = hidden_size
    
    def forward(self, x):
        # Implements Equation (1) from paper
        return x

class Architecture:
    """Architecture generator."""
    def generate(self):
        pass
''',
        "trainer.py": '''"""
Training loop implementation.
Implements Section 4.1 of the paper.
"""
def train(model, data):
    """Train the model using policy gradient."""
    # Learning rate from paper Section 4.1
    learning_rate = 0.001
    epochs = 100
    pass
''',
        "evaluation.py": '''"""
Evaluation metrics.
"""
def evaluate(model, test_data):
    """Evaluate model on test data."""
    pass
'''
    }


@pytest.fixture
def sample_planning_artifacts() -> Dict:
    """Sample planning artifacts for testing."""
    return {
        "logic_analysis": [
            ["model.py", "Defines the Controller and Architecture classes for neural architecture search"],
            ["trainer.py", "Implements the training loop with policy gradient methods"],
            ["evaluation.py", "Contains evaluation metrics and testing functions"]
        ],
        "task_list": ["model.py", "trainer.py", "evaluation.py"],
        "Implementation approach": "We will implement a controller-based neural architecture search system"
    }


@pytest.fixture
def sample_config_data() -> Dict:
    """Sample configuration data for testing."""
    return {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "hidden_size": 128,
        "required_packages": ["torch==1.9.0", "numpy==1.21.0"]
    }


@pytest.fixture
def sample_traceability_map() -> Dict:
    """Sample traceability map for testing."""
    return {
        "code_to_paper": {
            "model.py:Controller": ["Methodology", "Section 3.2"],
            "trainer.py:train": ["Experiments", "Section 4.1"],
            "evaluation.py:evaluate": ["Experiments"]
        },
        "paper_to_code": {
            "Methodology": [
                {"component": "model.py:Controller", "description": "Controller network", "file": "model.py"}
            ],
            "Experiments": [
                {"component": "trainer.py:train", "description": "Training loop", "file": "trainer.py"}
            ]
        },
        "paper_sections": [
            {"section": "Introduction", "text": "..."},
            {"section": "Methodology", "text": "..."},
            {"section": "Experiments", "text": "..."}
        ],
        "coverage_score": 0.67
    }


@pytest.fixture
def sample_missing_info() -> List[Dict]:
    """Sample missing information alerts for testing."""
    return [
        {
            "type": "hyperparameter",
            "parameter": "weight_decay",
            "description": "Hyperparameter 'weight_decay' is used in code but not explicitly specified in paper",
            "current_value": "0.0001",
            "severity": "medium",
            "suggestion": "Review paper or standard practices"
        },
        {
            "type": "dataset",
            "parameter": "dataset",
            "description": "Dataset loading code exists but dataset is not clearly specified in paper",
            "severity": "high",
            "suggestion": "Verify dataset compatibility with paper's experimental setup"
        }
    ]


@pytest.fixture
def sample_metrics() -> Dict:
    """Sample explainability metrics for testing."""
    return {
        "traceability_coverage": 0.67,
        "comment_density": 0.15,
        "paper_reference_accuracy": 0.75,
        "missing_info_score": 0.85,
        "readability_score": 0.70,
        "overall_explainability_score": 0.68
    }


@pytest.fixture
def temp_output_dir() -> Path:
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="ep2c_test_")
    yield Path(temp_dir)
    # Cleanup handled by pytest


@pytest.fixture
def sample_paper_content() -> str:
    """Sample paper content as string for testing."""
    return """
    This paper presents a novel approach to neural architecture search.
    Our method uses a controller network to generate architectures.
    We evaluate on CIFAR-10 and ImageNet datasets.
    We use a learning rate of 0.001 and train for 100 epochs.
    """

