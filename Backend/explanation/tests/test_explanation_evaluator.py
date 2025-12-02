"""
Tests for ExplanationEvaluator class.
Tests explainability metrics calculation and evaluation.
"""

import pytest
import sys
from pathlib import Path

# Add explanation directory to path
explanation_dir = Path(__file__).parent.parent
sys.path.insert(0, str(explanation_dir))

from explanation_evaluator import ExplanationEvaluator


class TestExplanationEvaluator:
    """Test suite for ExplanationEvaluator."""
    
    def test_init(self):
        """Test evaluator initialization."""
        evaluator = ExplanationEvaluator()
        assert evaluator is not None
    
    def test_calculate_traceability_coverage(self, sample_traceability_map):
        """Test traceability coverage calculation."""
        evaluator = ExplanationEvaluator()
        paper_sections = sample_traceability_map.get("paper_sections", [])
        
        coverage = evaluator.calculate_traceability_coverage(
            sample_traceability_map,
            paper_sections
        )
        
        # Should return coverage_score from map
        assert coverage == sample_traceability_map["coverage_score"]
        assert 0.0 <= coverage <= 1.0
    
    def test_calculate_traceability_coverage_no_score(self):
        """Test coverage calculation when score is missing."""
        evaluator = ExplanationEvaluator()
        traceability_map = {"code_to_paper": {}}
        coverage = evaluator.calculate_traceability_coverage(traceability_map, [])
        assert coverage == 0.0
    
    def test_calculate_comment_density(self, sample_generated_files):
        """Test comment density calculation."""
        evaluator = ExplanationEvaluator()
        density = evaluator.calculate_comment_density(sample_generated_files)
        
        # Should return a value between 0 and 1
        assert 0.0 <= density <= 1.0
        # Sample files have docstrings, so density should be > 0
        assert density > 0.0
    
    def test_calculate_comment_density_no_comments(self):
        """Test comment density with code that has no comments."""
        evaluator = ExplanationEvaluator()
        code_no_comments = {
            "file.py": "def func():\n    return x\nclass Test:\n    pass"
        }
        density = evaluator.calculate_comment_density(code_no_comments)
        assert density == 0.0
    
    def test_calculate_comment_density_empty(self):
        """Test comment density with empty code dictionary."""
        evaluator = ExplanationEvaluator()
        density = evaluator.calculate_comment_density({})
        assert density == 0.0
    
    def test_calculate_paper_reference_accuracy(self, sample_generated_files):
        """Test paper reference accuracy calculation."""
        evaluator = ExplanationEvaluator()
        accuracy = evaluator.calculate_paper_reference_accuracy(sample_generated_files)
        
        # Should return a value between 0 and 1
        assert 0.0 <= accuracy <= 1.0
        # Sample files mention "Section" and "paper", so accuracy should be > 0
        assert accuracy > 0.0
    
    def test_calculate_paper_reference_accuracy_no_references(self):
        """Test reference accuracy with code that has no paper references."""
        evaluator = ExplanationEvaluator()
        code_no_refs = {
            "file.py": "def func():\n    x = 1\n    return x"
        }
        accuracy = evaluator.calculate_paper_reference_accuracy(code_no_refs)
        assert accuracy == 0.0
    
    def test_calculate_paper_reference_accuracy_patterns(self):
        """Test that various reference patterns are detected."""
        evaluator = ExplanationEvaluator()
        
        # Test different reference patterns
        code_with_refs = {
            "file1.py": "# Implements Section 3.2",
            "file2.py": "# See Equation (5) from paper",
            "file3.py": "# Figure 1 shows the architecture",
            "file4.py": "# Table 2 contains the results",
            "file5.py": "# Algorithm 1 described in paper",
            "file6.py": "# This implements the method",
            "file7.py": "def func(): pass"  # No reference
        }
        accuracy = evaluator.calculate_paper_reference_accuracy(code_with_refs)
        # 6 out of 7 files have references
        assert accuracy == pytest.approx(6.0 / 7.0, abs=0.01)
    
    def test_calculate_missing_info_score_no_missing(self):
        """Test missing info score when no information is missing."""
        evaluator = ExplanationEvaluator()
        score = evaluator.calculate_missing_info_score([])
        assert score == 1.0  # Perfect score when nothing is missing
    
    def test_calculate_missing_info_score_with_missing(self, sample_missing_info):
        """Test missing info score calculation with missing items."""
        evaluator = ExplanationEvaluator()
        score = evaluator.calculate_missing_info_score(sample_missing_info)
        
        # Score should be between 0 and 1, lower when more items are missing
        assert 0.0 <= score <= 1.0
        assert score < 1.0  # Should be less than perfect
    
    def test_calculate_missing_info_score_severity_weights(self):
        """Test that severity weights affect the score correctly."""
        evaluator = ExplanationEvaluator()
        
        # High severity items should lower score more
        high_severity = [
            {"severity": "high", "parameter": "test1"},
            {"severity": "high", "parameter": "test2"},
            {"severity": "high", "parameter": "test3"}
        ]
        score_high = evaluator.calculate_missing_info_score(high_severity)
        
        # Low severity items should lower score less
        low_severity = [
            {"severity": "low", "parameter": "test1"},
            {"severity": "low", "parameter": "test2"},
            {"severity": "low", "parameter": "test3"}
        ]
        score_low = evaluator.calculate_missing_info_score(low_severity)
        
        # High severity should result in lower score
        assert score_high < score_low
    
    def test_calculate_readability_score(self, sample_generated_files):
        """Test readability score calculation."""
        evaluator = ExplanationEvaluator()
        score = evaluator.calculate_readability_score(sample_generated_files)
        
        # Should return a value between 0 and 1
        assert 0.0 <= score <= 1.0
        # Sample files have docstrings, so score should be > 0
        assert score > 0.0
    
    def test_calculate_readability_score_components(self):
        """Test that readability considers docstrings, type hints, and comments."""
        evaluator = ExplanationEvaluator()
        
        # Code with all readability features
        good_code = {
            "file.py": '''"""
Module docstring.
"""
def func(x: int) -> int:
    """Function docstring."""
    # Inline comment
    return x + 1
'''
        }
        score_good = evaluator.calculate_readability_score(good_code)
        
        # Code with minimal features
        bad_code = {
            "file.py": "def func(x):\n    return x+1"
        }
        score_bad = evaluator.calculate_readability_score(bad_code)
        
        # Good code should score higher
        assert score_good > score_bad
    
    def test_calculate_readability_score_empty(self):
        """Test readability score with empty code."""
        evaluator = ExplanationEvaluator()
        score = evaluator.calculate_readability_score({})
        assert score == 0.0
    
    def test_evaluate_explainability(self, sample_generated_files, 
                                     sample_traceability_map, sample_missing_info):
        """Test full explainability evaluation."""
        evaluator = ExplanationEvaluator()
        paper_sections = sample_traceability_map.get("paper_sections", [])
        
        metrics = evaluator.evaluate_explainability(
            sample_generated_files,
            sample_traceability_map,
            sample_missing_info,
            paper_sections
        )
        
        # Should return all expected metrics
        assert "traceability_coverage" in metrics
        assert "comment_density" in metrics
        assert "paper_reference_accuracy" in metrics
        assert "missing_info_score" in metrics
        assert "readability_score" in metrics
        assert "overall_explainability_score" in metrics
        
        # All scores should be between 0 and 1
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0
        
        # Overall score should be weighted average
        assert metrics["overall_explainability_score"] > 0.0
    
    def test_generate_explanation_report(self, sample_metrics):
        """Test generation of explanation evaluation report."""
        evaluator = ExplanationEvaluator()
        report = evaluator.generate_explanation_report(sample_metrics)
        
        # Should contain key sections
        assert "EXPLAINABILITY EVALUATION REPORT" in report
        assert "Overall Explainability Score" in report
        assert "Detailed Metrics" in report
        assert "Traceability Coverage" in report
        assert "Comment Density" in report
        assert "Interpretation" in report
        assert "Recommendations" in report
    
    def test_generate_explanation_report_interpretation(self):
        """Test that report includes appropriate interpretation."""
        evaluator = ExplanationEvaluator()
        
        # Test excellent score
        excellent_metrics = {
            "overall_explainability_score": 0.85,
            "traceability_coverage": 0.9,
            "comment_density": 0.3,
            "paper_reference_accuracy": 0.8,
            "missing_info_score": 0.9,
            "readability_score": 0.8
        }
        report = evaluator.generate_explanation_report(excellent_metrics)
        assert "Excellent" in report or "excellent" in report.lower()
        
        # Test low score
        low_metrics = {
            "overall_explainability_score": 0.3,
            "traceability_coverage": 0.2,
            "comment_density": 0.05,
            "paper_reference_accuracy": 0.1,
            "missing_info_score": 0.5,
            "readability_score": 0.2
        }
        report = evaluator.generate_explanation_report(low_metrics)
        assert "Low" in report or "low" in report.lower() or "improvements needed" in report.lower()
    
    def test_generate_explanation_report_recommendations(self):
        """Test that report includes recommendations based on metrics."""
        evaluator = ExplanationEvaluator()
        
        # Metrics with low traceability coverage
        metrics = {
            "overall_explainability_score": 0.5,
            "traceability_coverage": 0.3,  # Low
            "comment_density": 0.1,  # Low
            "paper_reference_accuracy": 0.4,  # Low
            "missing_info_score": 0.6,  # Low
            "readability_score": 0.4  # Low
        }
        report = evaluator.generate_explanation_report(metrics)
        
        # Should include recommendations for low metrics
        assert "Recommendations" in report
        assert "traceability" in report.lower() or "links" in report.lower()
        assert "comments" in report.lower() or "docstrings" in report.lower()

