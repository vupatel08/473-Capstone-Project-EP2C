"""
Tests for READMEGenerator class.
Tests README and EXPLANATION.md generation.
"""

import pytest
import sys
from pathlib import Path

# Add explanation directory to path
explanation_dir = Path(__file__).parent.parent
sys.path.insert(0, str(explanation_dir))

from readme_generator import READMEGenerator


class TestREADMEGenerator:
    """Test suite for READMEGenerator."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = READMEGenerator()
        assert generator is not None
    
    def test_generate_header(self):
        """Test README header generation."""
        generator = READMEGenerator()
        paper_metadata = {
            "title": "Test Paper",
            "authors": ["Author One", "Author Two"],
            "url": "https://example.com",
            "abstract": "Test abstract"
        }
        
        header = generator._generate_header(paper_metadata)
        
        # Should contain paper information
        assert "Test Paper" in header
        assert "Author One" in header
        assert "Author Two" in header
        assert "https://example.com" in header
        assert "EP2C" in header
    
    def test_generate_header_missing_fields(self):
        """Test header generation with missing metadata fields."""
        generator = READMEGenerator()
        paper_metadata = {}  # Empty metadata
        
        header = generator._generate_header(paper_metadata)
        
        # Should handle missing fields gracefully
        assert "Implementation" in header or "Unknown" in header
        assert "EP2C" in header
    
    def test_generate_overview(self):
        """Test overview section generation."""
        generator = READMEGenerator()
        overview = generator._generate_overview({})
        
        # Should contain key features
        assert "Overview" in overview
        assert "traceability" in overview.lower() or "Traceability" in overview
        assert "Implementation" in overview
    
    def test_generate_requirements_with_config(self, sample_config_data):
        """Test requirements section with config data."""
        generator = READMEGenerator()
        requirements = generator._generate_requirements({}, sample_config_data)
        
        # Should include packages from config
        assert "Requirements" in requirements
        assert "torch" in requirements.lower()
        assert "numpy" in requirements.lower()
    
    def test_generate_requirements_without_config(self):
        """Test requirements section without config."""
        generator = READMEGenerator()
        requirements = generator._generate_requirements({}, None)
        
        # Should include default packages
        assert "Requirements" in requirements
        assert "torch" in requirements.lower() or "Install" in requirements
    
    def test_generate_structure_with_task_list(self):
        """Test structure section with task list."""
        generator = READMEGenerator()
        code_structure = {
            "task_list": ["model.py", "trainer.py", "evaluation.py"]
        }
        
        structure = generator._generate_structure(code_structure)
        
        # Should include files from task list
        assert "Repository Structure" in structure
        assert "model.py" in structure
        assert "trainer.py" in structure
        assert "evaluation.py" in structure
    
    def test_generate_structure_without_task_list(self):
        """Test structure section without task list."""
        generator = READMEGenerator()
        structure = generator._generate_structure({})
        
        # Should include default file structure
        assert "Repository Structure" in structure
        assert "model.py" in structure or "main.py" in structure
    
    def test_generate_code_to_paper_mapping(self, sample_traceability_map):
        """Test code-to-paper mapping table generation."""
        generator = READMEGenerator()
        mapping = generator._generate_code_to_paper_mapping(sample_traceability_map)
        
        # Should contain mapping table
        assert "Code-to-Paper Traceability" in mapping
        assert "Traceability Coverage" in mapping
        # Should include coverage score
        assert "%" in mapping
    
    def test_generate_code_to_paper_mapping_empty(self):
        """Test mapping generation with empty traceability map."""
        generator = READMEGenerator()
        empty_map = {"code_to_paper": {}, "coverage_score": 0.0}
        mapping = generator._generate_code_to_paper_mapping(empty_map)
        
        # Should still generate section
        assert "Code-to-Paper Traceability" in mapping
    
    def test_generate_missing_information_with_items(self, sample_missing_info):
        """Test missing information section with items."""
        generator = READMEGenerator()
        missing = generator._generate_missing_information(sample_missing_info)
        
        # Should contain missing information section
        assert "Missing Information" in missing
        # Should list missing items
        assert "hyperparameter" in missing.lower() or "dataset" in missing.lower()
    
    def test_generate_missing_information_empty(self):
        """Test missing information section with no missing items."""
        generator = READMEGenerator()
        missing = generator._generate_missing_information([])
        
        # Should indicate no missing information
        assert "Missing Information" in missing
        assert "No critical missing information" in missing or "detected" in missing.lower()
    
    def test_generate_getting_started(self):
        """Test getting started section generation."""
        generator = READMEGenerator()
        getting_started = generator._generate_getting_started({}, None)
        
        # Should contain getting started instructions
        assert "Getting Started" in getting_started
        assert "Install" in getting_started or "install" in getting_started.lower()
        assert "python" in getting_started.lower() or "main.py" in getting_started
    
    def test_generate_next_steps_with_missing_info(self, sample_missing_info):
        """Test next steps section with missing information."""
        generator = READMEGenerator()
        next_steps = generator._generate_next_steps(sample_missing_info)
        
        # Should include actions for missing info
        assert "Next Steps" in next_steps
        assert "Missing Information" in next_steps or "missing" in next_steps.lower()
    
    def test_generate_next_steps_no_missing_info(self):
        """Test next steps section without missing information."""
        generator = READMEGenerator()
        next_steps = generator._generate_next_steps([])
        
        # Should still include recommended workflow
        assert "Next Steps" in next_steps
        assert "Workflow" in next_steps or "workflow" in next_steps.lower()
    
    def test_generate_acknowledgments(self):
        """Test acknowledgments section generation."""
        generator = READMEGenerator()
        paper_metadata = {
            "title": "Test Paper",
            "authors": ["Author One"]
        }
        
        acknowledgments = generator._generate_acknowledgments(paper_metadata)
        
        # Should contain acknowledgments
        assert "Acknowledgments" in acknowledgments
        assert "EP2C" in acknowledgments
        assert "Test Paper" in acknowledgments
        assert "Author One" in acknowledgments
    
    def test_generate_readme_full(self, sample_paper_json, sample_traceability_map, 
                                  sample_missing_info, sample_config_data):
        """Test full README generation."""
        generator = READMEGenerator()
        paper_metadata = {
            "title": sample_paper_json["title"],
            "authors": sample_paper_json["authors"],
            "url": sample_paper_json["url"],
            "abstract": sample_paper_json["abstract"]
        }
        code_structure = {"task_list": ["model.py", "trainer.py"]}
        
        readme = generator.generate_readme(
            paper_metadata,
            code_structure,
            sample_traceability_map,
            sample_missing_info,
            sample_config_data
        )
        
        # Should contain all major sections
        assert "Test Paper" in readme or sample_paper_json["title"] in readme
        assert "Overview" in readme
        assert "Requirements" in readme
        assert "Repository Structure" in readme
        assert "Code-to-Paper Traceability" in readme
        assert "Missing Information" in readme
        assert "Getting Started" in readme
        assert "Next Steps" in readme
        assert "Acknowledgments" in readme
    
    def test_generate_comprehensive_explanation(self, sample_paper_json, sample_traceability_map,
                                                sample_missing_info, sample_metrics, 
                                                sample_config_data):
        """Test comprehensive EXPLANATION.md generation."""
        generator = READMEGenerator()
        paper_metadata = {
            "title": sample_paper_json["title"],
            "authors": sample_paper_json["authors"],
            "url": sample_paper_json["url"],
            "abstract": sample_paper_json["abstract"]
        }
        code_structure = {"task_list": ["model.py"]}
        
        explanation = generator.generate_comprehensive_explanation(
            paper_metadata,
            code_structure,
            sample_traceability_map,
            sample_missing_info,
            sample_metrics,
            sample_config_data
        )
        
        # Should contain all major sections
        assert "Explanation Layer" in explanation
        assert "Explainability Metrics" in explanation
        assert "Overall Explainability Score" in explanation
        assert "Code-to-Paper Traceability" in explanation
        assert "Missing Information" in explanation
        # Should include JSON data
        assert "traceability_map" in explanation.lower() or "Traceability Map" in explanation
        assert "metrics" in explanation.lower() or "Metrics" in explanation
    
    def test_generate_comprehensive_explanation_metrics_display(self, sample_metrics):
        """Test that metrics are properly displayed in comprehensive explanation."""
        generator = READMEGenerator()
        paper_metadata = {"title": "Test", "authors": [], "url": "", "abstract": ""}
        
        explanation = generator.generate_comprehensive_explanation(
            paper_metadata,
            {},
            {"code_to_paper": {}, "coverage_score": 0.5},
            [],
            sample_metrics,
            None
        )
        
        # Should display all metrics
        assert str(int(sample_metrics["overall_explainability_score"] * 100)) in explanation
        assert "Traceability Coverage" in explanation
        assert "Comment Density" in explanation
    
    def test_generate_comprehensive_explanation_severity_groups(self, sample_missing_info):
        """Test that missing info is grouped by severity in comprehensive explanation."""
        generator = READMEGenerator()
        paper_metadata = {"title": "Test", "authors": [], "url": "", "abstract": ""}
        
        explanation = generator.generate_comprehensive_explanation(
            paper_metadata,
            {},
            {"code_to_paper": {}, "coverage_score": 0.0},
            sample_missing_info,
            {"overall_explainability_score": 0.5, "traceability_coverage": 0.0,
             "comment_density": 0.0, "paper_reference_accuracy": 0.0,
             "missing_info_score": 0.5, "readability_score": 0.0},
            None
        )
        
        # Should group by severity
        assert "High Priority" in explanation or "high" in explanation.lower()
        assert "Medium Priority" in explanation or "medium" in explanation.lower()

