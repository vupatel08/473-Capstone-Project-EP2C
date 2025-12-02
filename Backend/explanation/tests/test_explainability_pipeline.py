"""
Tests for ExplainabilityPipeline class.
Tests full pipeline integration and file generation.
"""

import pytest
import sys
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add explanation directory to path
explanation_dir = Path(__file__).parent.parent
sys.path.insert(0, str(explanation_dir))

from explainability_pipeline import ExplainabilityPipeline


class TestExplainabilityPipeline:
    """Test suite for ExplainabilityPipeline."""
    
    def test_init(self):
        """Test pipeline initialization."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            pipeline = ExplainabilityPipeline()
            assert pipeline.explanation_generator is not None
            assert pipeline.readme_generator is not None
            assert pipeline.missing_info_detector is not None
            assert pipeline.evaluator is not None
    
    def test_init_with_api_key(self):
        """Test initialization with custom API key."""
        pipeline = ExplainabilityPipeline(openai_api_key='custom-key')
        assert pipeline.explanation_generator is not None
    
    def test_load_json(self, temp_output_dir, sample_paper_json):
        """Test JSON file loading."""
        # Create test JSON file
        json_path = temp_output_dir / "test.json"
        with open(json_path, 'w') as f:
            json.dump(sample_paper_json, f)
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            pipeline = ExplainabilityPipeline()
            loaded = pipeline._load_json(str(json_path))
            
            assert loaded["title"] == sample_paper_json["title"]
            assert loaded["authors"] == sample_paper_json["authors"]
    
    def test_load_yaml(self, temp_output_dir):
        """Test YAML file loading."""
        import yaml
        yaml_path = temp_output_dir / "test.yaml"
        yaml_data = {"learning_rate": 0.001, "batch_size": 32}
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            pipeline = ExplainabilityPipeline()
            loaded = pipeline._load_yaml(str(yaml_path))
            
            assert loaded["learning_rate"] == 0.001
            assert loaded["batch_size"] == 32
    
    def test_load_generated_files(self, temp_output_dir, sample_generated_files):
        """Test loading generated code files from directory."""
        # Create code directory structure
        code_dir = temp_output_dir / "code"
        code_dir.mkdir()
        
        for file_path, content in sample_generated_files.items():
            file_full_path = code_dir / file_path
            file_full_path.parent.mkdir(parents=True, exist_ok=True)
            file_full_path.write_text(content)
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            pipeline = ExplainabilityPipeline()
            loaded_files = pipeline._load_generated_files(str(code_dir))
            
            # Should load all Python files
            assert len(loaded_files) == len(sample_generated_files)
            for file_path in sample_generated_files.keys():
                assert file_path in loaded_files or any(file_path in k for k in loaded_files.keys())
    
    def test_load_generated_files_ignores_non_python(self, temp_output_dir):
        """Test that non-Python files are ignored."""
        code_dir = temp_output_dir / "code"
        code_dir.mkdir()
        
        # Create Python and non-Python files
        (code_dir / "file.py").write_text("def func(): pass")
        (code_dir / "file.txt").write_text("text content")
        (code_dir / "file.json").write_text('{"data": "test"}')
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            pipeline = ExplainabilityPipeline()
            loaded_files = pipeline._load_generated_files(str(code_dir))
            
            # Should only load Python files
            assert "file.py" in loaded_files or any("file.py" in k for k in loaded_files.keys())
            assert "file.txt" not in str(loaded_files)
            assert "file.json" not in str(loaded_files)
    
    def test_extract_paper_content(self, sample_paper_json):
        """Test paper content extraction."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            pipeline = ExplainabilityPipeline()
            content = pipeline._extract_paper_content(sample_paper_json)
            
            # Should combine abstract and body text
            assert sample_paper_json["abstract"] in content
            assert "Neural architecture search" in content
    
    def test_extract_paper_metadata(self, sample_paper_json):
        """Test paper metadata extraction."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            pipeline = ExplainabilityPipeline()
            metadata = pipeline._extract_paper_metadata(sample_paper_json)
            
            assert metadata["title"] == sample_paper_json["title"]
            assert metadata["authors"] == sample_paper_json["authors"]
            assert metadata["url"] == sample_paper_json["url"]
            assert metadata["abstract"] == sample_paper_json["abstract"]
    
    def test_extract_paper_metadata_missing_fields(self):
        """Test metadata extraction with missing fields."""
        paper_json = {"title": "Test"}  # Missing other fields
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            pipeline = ExplainabilityPipeline()
            metadata = pipeline._extract_paper_metadata(paper_json)
            
            assert metadata["title"] == "Test"
            assert metadata["authors"] == []  # Default empty list
            assert metadata["url"] == ""  # Default empty string
    
    @patch('explainability_pipeline.ExplanationGenerator')
    @patch('explainability_pipeline.MissingInfoDetector')
    @patch('explainability_pipeline.READMEGenerator')
    @patch('explainability_pipeline.ExplanationEvaluator')
    def test_generate_explanation_layer_full(self, mock_evaluator, mock_readme, 
                                             mock_detector, mock_generator,
                                             temp_output_dir, sample_paper_json,
                                             sample_generated_files, sample_planning_artifacts):
        """Test full explanation layer generation."""
        # Setup mocks
        mock_gen_instance = MagicMock()
        mock_gen_instance.generate_traceability_map.return_value = {
            "code_to_paper": {},
            "paper_to_code": {},
            "paper_sections": [],
            "coverage_score": 0.5
        }
        mock_generator.return_value = mock_gen_instance
        
        mock_det_instance = MagicMock()
        mock_det_instance.detect_missing_information.return_value = []
        mock_detector.return_value = mock_det_instance
        
        mock_readme_instance = MagicMock()
        mock_readme_instance.generate_readme.return_value = "# README"
        mock_readme_instance.generate_comprehensive_explanation.return_value = "# EXPLANATION"
        mock_readme.return_value = mock_readme_instance
        
        mock_eval_instance = MagicMock()
        mock_eval_instance.evaluate_explainability.return_value = {
            "overall_explainability_score": 0.7
        }
        mock_eval_instance.generate_explanation_report.return_value = "Report"
        mock_evaluator.return_value = mock_eval_instance
        
        # Create input files
        paper_json_path = temp_output_dir / "paper.json"
        with open(paper_json_path, 'w') as f:
            json.dump(sample_paper_json, f)
        
        code_dir = temp_output_dir / "code"
        code_dir.mkdir()
        for file_path, content in sample_generated_files.items():
            (code_dir / file_path).write_text(content)
        
        planning_path = temp_output_dir / "planning.json"
        with open(planning_path, 'w') as f:
            json.dump(sample_planning_artifacts, f)
        
        output_dir = temp_output_dir / "explanation_output"
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            pipeline = ExplainabilityPipeline()
            result = pipeline.generate_explanation_layer(
                str(paper_json_path),
                str(code_dir),
                str(planning_path),
                str(output_dir),
                None
            )
            
            # Should return result dictionary
            assert "traceability_map" in result
            assert "missing_info" in result
            assert "readme" in result
            assert "comprehensive_explanation" in result
            assert "metrics" in result
            assert "report" in result
            
            # Should create output files
            assert (output_dir / "traceability_map.json").exists()
            assert (output_dir / "README.md").exists()
            assert (output_dir / "EXPLANATION.md").exists()
            assert (output_dir / "missing_information.json").exists()
            assert (output_dir / "explainability_metrics.json").exists()
            assert (output_dir / "explainability_report.txt").exists()
    
    def test_generate_explanation_layer_with_config(self, temp_output_dir, sample_paper_json,
                                                    sample_generated_files, sample_planning_artifacts,
                                                    sample_config_data):
        """Test explanation layer generation with config file."""
        import yaml
        
        # Create input files
        paper_json_path = temp_output_dir / "paper.json"
        with open(paper_json_path, 'w') as f:
            json.dump(sample_paper_json, f)
        
        code_dir = temp_output_dir / "code"
        code_dir.mkdir()
        for file_path, content in sample_generated_files.items():
            (code_dir / file_path).write_text(content)
        
        planning_path = temp_output_dir / "planning.json"
        with open(planning_path, 'w') as f:
            json.dump(sample_planning_artifacts, f)
        
        config_path = temp_output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(sample_config_data, f)
        
        output_dir = temp_output_dir / "explanation_output"
        
        # Mock the components to avoid API calls
        with patch('explainability_pipeline.ExplanationGenerator') as mock_gen_class, \
             patch('explainability_pipeline.MissingInfoDetector') as mock_det_class, \
             patch('explainability_pipeline.READMEGenerator') as mock_readme_class, \
             patch('explainability_pipeline.ExplanationEvaluator') as mock_eval_class, \
             patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            
            # Setup mock instances
            mock_gen = MagicMock()
            mock_gen.generate_traceability_map.return_value = {
                "code_to_paper": {}, "paper_to_code": {}, 
                "paper_sections": [], "coverage_score": 0.5
            }
            mock_gen_class.return_value = mock_gen
            
            mock_det = MagicMock()
            mock_det.detect_missing_information.return_value = []
            mock_det_class.return_value = mock_det
            
            mock_readme = MagicMock()
            mock_readme.generate_readme.return_value = "# README"
            mock_readme.generate_comprehensive_explanation.return_value = "# EXPLANATION"
            mock_readme_class.return_value = mock_readme
            
            mock_eval = MagicMock()
            mock_eval.evaluate_explainability.return_value = {"overall_explainability_score": 0.7}
            mock_eval.generate_explanation_report.return_value = "Report"
            mock_eval_class.return_value = mock_eval
            
            pipeline = ExplainabilityPipeline()
            result = pipeline.generate_explanation_layer(
                str(paper_json_path),
                str(code_dir),
                str(planning_path),
                str(output_dir),
                str(config_path)
            )
            
            # Should successfully generate with config
            assert result is not None
            assert "metrics" in result
    
    def test_generate_explanation_layer_file_creation(self, temp_output_dir, sample_paper_json,
                                                      sample_generated_files, sample_planning_artifacts):
        """Test that all expected files are created."""
        # Create input files
        paper_json_path = temp_output_dir / "paper.json"
        with open(paper_json_path, 'w') as f:
            json.dump(sample_paper_json, f)
        
        code_dir = temp_output_dir / "code"
        code_dir.mkdir()
        for file_path, content in sample_generated_files.items():
            (code_dir / file_path).write_text(content)
        
        planning_path = temp_output_dir / "planning.json"
        with open(planning_path, 'w') as f:
            json.dump(sample_planning_artifacts, f)
        
        output_dir = temp_output_dir / "explanation_output"
        
        # Mock all components
        with patch('explainability_pipeline.ExplanationGenerator') as mock_gen_class, \
             patch('explainability_pipeline.MissingInfoDetector') as mock_det_class, \
             patch('explainability_pipeline.READMEGenerator') as mock_readme_class, \
             patch('explainability_pipeline.ExplanationEvaluator') as mock_eval_class, \
             patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            
            mock_gen = MagicMock()
            mock_gen.generate_traceability_map.return_value = {
                "code_to_paper": {}, "paper_to_code": {},
                "paper_sections": [], "coverage_score": 0.5
            }
            mock_gen_class.return_value = mock_gen
            
            mock_det = MagicMock()
            mock_det.detect_missing_information.return_value = []
            mock_det_class.return_value = mock_det
            
            mock_readme = MagicMock()
            mock_readme.generate_readme.return_value = "# README content"
            mock_readme.generate_comprehensive_explanation.return_value = "# EXPLANATION content"
            mock_readme_class.return_value = mock_readme
            
            mock_eval = MagicMock()
            mock_eval.evaluate_explainability.return_value = {
                "overall_explainability_score": 0.7,
                "traceability_coverage": 0.5
            }
            mock_eval.generate_explanation_report.return_value = "Evaluation Report"
            mock_eval_class.return_value = mock_eval
            
            pipeline = ExplainabilityPipeline()
            pipeline.generate_explanation_layer(
                str(paper_json_path),
                str(code_dir),
                str(planning_path),
                str(output_dir),
                None
            )
            
            # Verify all expected files were created
            expected_files = [
                "traceability_map.json",
                "README.md",
                "EXPLANATION.md",
                "missing_information.json",
                "explainability_metrics.json",
                "explainability_report.txt"
            ]
            
            for filename in expected_files:
                file_path = output_dir / filename
                assert file_path.exists(), f"Expected file {filename} was not created"
                
                # Verify files have content
                if filename.endswith('.json'):
                    with open(file_path) as f:
                        data = json.load(f)
                        assert data is not None
                else:
                    content = file_path.read_text()
                    assert len(content) > 0

