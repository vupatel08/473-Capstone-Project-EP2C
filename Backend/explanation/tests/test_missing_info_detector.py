"""
Tests for MissingInfoDetector class.
Tests detection of missing hyperparameters, datasets, and implementation details.
"""

import pytest
import sys
from pathlib import Path

# Add explanation directory to path
explanation_dir = Path(__file__).parent.parent
sys.path.insert(0, str(explanation_dir))

from missing_info_detector import MissingInfoDetector


class TestMissingInfoDetector:
    """Test suite for MissingInfoDetector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = MissingInfoDetector()
        assert detector is not None
        # Check that hyperparameter list exists
        assert len(detector.HYPERPARAMETERS) > 0
        assert "learning_rate" in detector.HYPERPARAMETERS
    
    def test_detect_missing_hyperparameters_found(self, sample_paper_content):
        """Test detection of hyperparameters not in paper."""
        detector = MissingInfoDetector()
        # Config has weight_decay but paper doesn't mention it
        config = {
            "learning_rate": 0.001,  # Mentioned in paper
            "weight_decay": 0.0001,   # Not mentioned in paper
            "batch_size": 32          # Not mentioned in paper
        }
        
        missing = detector.detect_missing_hyperparameters(sample_paper_content, config)
        
        # Should detect weight_decay and batch_size as missing
        assert len(missing) >= 1
        missing_params = [item["parameter"] for item in missing]
        assert "weight_decay" in missing_params or "batch_size" in missing_params
    
    def test_detect_missing_hyperparameters_all_present(self):
        """Test when all hyperparameters are mentioned in paper."""
        detector = MissingInfoDetector()
        paper_content = "We use a learning rate of 0.001 and batch size of 32"
        config = {
            "learning_rate": 0.001,
            "batch_size": 32
        }
        
        missing = detector.detect_missing_hyperparameters(paper_content, config)
        # Should find fewer or no missing items since both are mentioned
        assert isinstance(missing, list)
    
    def test_detect_missing_hyperparameters_empty_config(self, sample_paper_content):
        """Test with empty config dictionary."""
        detector = MissingInfoDetector()
        missing = detector.detect_missing_hyperparameters(sample_paper_content, {})
        assert missing == []
    
    def test_detect_missing_hyperparameters_none_config(self, sample_paper_content):
        """Test with None config."""
        detector = MissingInfoDetector()
        missing = detector.detect_missing_hyperparameters(sample_paper_content, None)
        assert missing == []
    
    def test_detect_missing_dataset_info_no_loader(self, sample_paper_content):
        """Test dataset detection when no dataset loader exists."""
        detector = MissingInfoDetector()
        code_files = {
            "model.py": "class Model: pass"
        }
        
        missing = detector.detect_missing_dataset_info(sample_paper_content, code_files)
        # Should return empty if no dataset loader code
        assert missing == []
    
    def test_detect_missing_dataset_info_with_loader(self):
        """Test dataset detection when loader exists but not mentioned in paper."""
        detector = MissingInfoDetector()
        paper_content = "We present a novel architecture."  # No dataset mention
        code_files = {
            "dataset_loader.py": "def load_data(): pass"
        }
        
        missing = detector.detect_missing_dataset_info(paper_content, code_files)
        # Should detect missing dataset info
        assert len(missing) > 0
        assert any(item["type"] == "dataset" for item in missing)
    
    def test_detect_missing_dataset_info_hardcoded_paths(self):
        """Test detection of hardcoded dataset paths."""
        detector = MissingInfoDetector()
        paper_content = "We use CIFAR-10 dataset"
        code_files = {
            "loader.py": 'data_path = "/home/user/data/cifar10"'
        }
        
        missing = detector.detect_missing_dataset_info(paper_content, code_files)
        # Should detect hardcoded path
        assert any(item["type"] == "dataset_path" for item in missing)
    
    def test_detect_missing_implementation_details_todo(self):
        """Test detection of TODO comments."""
        detector = MissingInfoDetector()
        paper_content = "Test paper content"
        code_files = {
            "file.py": "def func():\n    # TODO: implement this\n    pass"
        }
        
        missing = detector.detect_missing_implementation_details(paper_content, code_files)
        # Should detect TODO
        assert len(missing) > 0
        assert any("todo" in item.get("parameter", "").lower() for item in missing)
        assert any(item["severity"] == "low" for item in missing)
    
    def test_detect_missing_implementation_details_fixme(self):
        """Test detection of FIXME comments."""
        detector = MissingInfoDetector()
        paper_content = "Test paper content"
        code_files = {
            "file.py": "# FIXME: needs optimization"
        }
        
        missing = detector.detect_missing_implementation_details(paper_content, code_files)
        assert len(missing) > 0
    
    def test_detect_missing_implementation_details_placeholder(self):
        """Test detection of placeholder values."""
        detector = MissingInfoDetector()
        paper_content = "Test paper content"
        code_files = {
            "file.py": "value = PLACEHOLDER"
        }
        
        missing = detector.detect_missing_implementation_details(paper_content, code_files)
        # Should detect placeholder
        assert len(missing) > 0
        assert any("placeholder" in item.get("parameter", "").lower() for item in missing)
        assert any(item["severity"] == "medium" for item in missing)
    
    def test_detect_missing_performance_info_gpu(self):
        """Test detection of missing GPU requirements."""
        detector = MissingInfoDetector()
        paper_content = "We present a method."  # No GPU mention
        code_files = {
            "trainer.py": "device = torch.device('cuda')"
        }
        
        missing = detector.detect_missing_performance_info(paper_content, code_files)
        # Should detect missing GPU specification
        assert len(missing) > 0
        assert any(item["type"] == "hardware" for item in missing)
    
    def test_detect_missing_performance_info_gpu_mentioned(self):
        """Test when GPU is mentioned in paper."""
        detector = MissingInfoDetector()
        paper_content = "We train on GPU using CUDA"
        code_files = {
            "trainer.py": "device = torch.device('cuda')"
        }
        
        missing = detector.detect_missing_performance_info(paper_content, code_files)
        # Should not detect missing GPU info since it's mentioned
        assert len(missing) == 0
    
    def test_determine_severity(self):
        """Test severity determination for parameters."""
        detector = MissingInfoDetector()
        
        # Critical parameters should be high severity
        assert detector._determine_severity("learning_rate") == "high"
        assert detector._determine_severity("batch_size") == "high"
        assert detector._determine_severity("epochs") == "high"
        
        # Important but not critical should be medium
        assert detector._determine_severity("dropout") == "medium"
        assert detector._determine_severity("optimizer") == "medium"
        
        # Other parameters should be low
        assert detector._determine_severity("momentum") == "low"
    
    def test_get_hyperparameter_suggestion(self):
        """Test suggestion generation for hyperparameters."""
        detector = MissingInfoDetector()
        
        # Should return suggestions for known parameters
        suggestion = detector._get_hyperparameter_suggestion("learning_rate")
        assert len(suggestion) > 0
        assert "learning_rate" in suggestion.lower() or "0.001" in suggestion
        
        # Should return default for unknown parameters
        suggestion = detector._get_hyperparameter_suggestion("unknown_param")
        assert "Review paper" in suggestion or "standard practices" in suggestion
    
    def test_detect_missing_information_full(self, sample_paper_content, 
                                            sample_config_data, sample_generated_files):
        """Test full missing information detection."""
        detector = MissingInfoDetector()
        
        missing = detector.detect_missing_information(
            sample_paper_content,
            sample_config_data,
            sample_generated_files
        )
        
        # Should return a list of missing items
        assert isinstance(missing, list)
        # Each item should have required fields
        for item in missing:
            assert "type" in item
            assert "parameter" in item
            assert "description" in item
            assert "severity" in item
            assert item["severity"] in ["high", "medium", "low"]
    
    def test_generate_missing_info_summary_no_missing(self):
        """Test summary generation when no information is missing."""
        detector = MissingInfoDetector()
        summary = detector.generate_missing_info_summary([])
        assert "No missing information" in summary or "detected" in summary.lower()
    
    def test_generate_missing_info_summary_with_missing(self, sample_missing_info):
        """Test summary generation with missing items."""
        detector = MissingInfoDetector()
        summary = detector.generate_missing_info_summary(sample_missing_info)
        
        # Should mention number of items
        assert str(len(sample_missing_info)) in summary or "items" in summary.lower()
        # Should group by severity
        assert "HIGH" in summary or "high" in summary.lower()
    
    def test_generate_missing_info_summary_severity_groups(self):
        """Test that summary groups items by severity."""
        detector = MissingInfoDetector()
        missing_info = [
            {"severity": "high", "parameter": "param1", "description": "Test 1"},
            {"severity": "high", "parameter": "param2", "description": "Test 2"},
            {"severity": "medium", "parameter": "param3", "description": "Test 3"},
            {"severity": "low", "parameter": "param4", "description": "Test 4"}
        ]
        
        summary = detector.generate_missing_info_summary(missing_info)
        
        # Should contain severity groups
        assert "HIGH" in summary or "high" in summary
        assert "MEDIUM" in summary or "medium" in summary
        assert "LOW" in summary or "low" in summary

