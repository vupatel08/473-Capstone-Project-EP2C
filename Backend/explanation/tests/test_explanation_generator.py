"""
Tests for ExplanationGenerator class.
Tests traceability map generation and code-paper linking.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add explanation directory to path
explanation_dir = Path(__file__).parent.parent
sys.path.insert(0, str(explanation_dir))

from explanation_generator import ExplanationGenerator


class TestExplanationGenerator:
    """Test suite for ExplanationGenerator."""
    
    def test_init_with_api_key(self):
        """Test initialization with provided API key."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            generator = ExplanationGenerator(openai_api_key='custom-key')
            assert generator.client is not None
    
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                ExplanationGenerator()
    
    def test_extract_paper_sections(self, sample_paper_json):
        """Test extraction of paper sections from JSON."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            generator = ExplanationGenerator()
            sections = generator._extract_paper_sections(sample_paper_json)
            
            # Should extract abstract and body sections
            assert len(sections) == 4  # Abstract + 3 body sections
            assert any(s['section'] == 'Abstract' for s in sections)
            assert any(s['section'] == 'Introduction' for s in sections)
            assert any(s['section'] == 'Methodology' for s in sections)
    
    def test_extract_paper_sections_no_abstract(self):
        """Test section extraction when abstract is missing."""
        paper_json = {
            "body_text": [{"section": "Introduction", "text": "Test"}]
        }
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            generator = ExplanationGenerator()
            sections = generator._extract_paper_sections(paper_json)
            assert len(sections) == 1
            assert sections[0]['section'] == 'Introduction'
    
    def test_extract_code_components(self, sample_generated_files):
        """Test extraction of code components (classes, functions)."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            generator = ExplanationGenerator()
            code_content = sample_generated_files['model.py']
            components = generator._extract_code_components(code_content, 'model.py')
            
            # Should extract Controller and Architecture classes
            assert 'model.py:Controller' in components
            assert 'model.py:Architecture' in components
            assert components['model.py:Controller']['type'] == 'class'
            assert 'forward' in components['model.py:Controller']['methods']
    
    def test_extract_code_components_functions(self, sample_generated_files):
        """Test extraction of standalone functions."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            generator = ExplanationGenerator()
            code_content = sample_generated_files['trainer.py']
            components = generator._extract_code_components(code_content, 'trainer.py')
            
            # Should extract train function
            assert 'trainer.py:train' in components
            assert components['trainer.py:train']['type'] == 'function'
    
    def test_extract_docstring(self):
        """Test docstring extraction from code blocks."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            generator = ExplanationGenerator()
            
            # Test triple-quoted docstring
            code_with_doc = '"""This is a docstring."""\ndef func(): pass'
            docstring = generator._extract_docstring(code_with_doc)
            assert docstring == "This is a docstring."
            
            # Test single-quoted docstring
            code_with_single = "'''Another docstring.'''\nclass Test: pass"
            docstring = generator._extract_docstring(code_with_single)
            assert docstring == "Another docstring."
            
            # Test no docstring
            code_no_doc = "def func(): pass"
            docstring = generator._extract_docstring(code_no_doc)
            assert docstring == ""
    
    def test_calculate_coverage_score(self):
        """Test traceability coverage score calculation."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            generator = ExplanationGenerator()
            
            # Test with mappings
            code_to_paper = {
                "file1:Class1": ["Section 1", "Section 2"],
                "file2:func1": ["Section 1"]
            }
            paper_sections = [
                {"section": "Section 1"},
                {"section": "Section 2"},
                {"section": "Section 3"}
            ]
            score = generator._calculate_coverage_score(code_to_paper, paper_sections)
            # 2 unique sections covered out of 3 = 0.667
            assert score == pytest.approx(0.667, abs=0.01)
            
            # Test with no sections
            score = generator._calculate_coverage_score({}, [])
            assert score == 0.0
    
    @patch('explanation_generator.OpenAI')
    def test_find_related_paper_sections(self, mock_openai, sample_planning_artifacts):
        """Test finding related paper sections using LLM."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Methodology, Section 3.2"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            generator = ExplanationGenerator()
            generator.client = mock_client
            
            component_info = {
                "type": "class",
                "description": "Controller network"
            }
            paper_sections = [
                {"section": "Introduction", "text": "..."},
                {"section": "Methodology", "text": "..."}
            ]
            
            sections = generator._find_related_paper_sections(
                "model.py",
                component_info,
                sample_planning_artifacts,
                paper_sections
            )
            
            assert len(sections) > 0
            assert "Methodology" in sections or "Section 3.2" in sections
    
    @patch('explanation_generator.OpenAI')
    def test_generate_traceability_map(self, mock_openai, sample_paper_json, 
                                       sample_generated_files, sample_planning_artifacts):
        """Test full traceability map generation."""
        # Mock OpenAI response for section finding
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Methodology, Experiments"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            generator = ExplanationGenerator()
            generator.client = mock_client
            
            traceability_map = generator.generate_traceability_map(
                sample_paper_json,
                sample_generated_files,
                sample_planning_artifacts
            )
            
            # Should have code_to_paper and paper_to_code mappings
            assert "code_to_paper" in traceability_map
            assert "paper_to_code" in traceability_map
            assert "paper_sections" in traceability_map
            assert "coverage_score" in traceability_map
            assert isinstance(traceability_map["coverage_score"], float)
    
    def test_generate_explanation_summary(self, sample_traceability_map):
        """Test generation of human-readable explanation summary."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            generator = ExplanationGenerator()
            summary = generator.generate_explanation_summary(sample_traceability_map)
            
            assert "EXPLANATION LAYER SUMMARY" in summary
            assert "Traceability Coverage" in summary
            assert "Code Components Mapped" in summary
            assert "Paper Sections Covered" in summary
    
    def test_format_paper_sections_for_prompt(self):
        """Test formatting of paper sections for LLM prompt."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            generator = ExplanationGenerator()
            sections = [
                {"section": "Introduction", "text": "This is a long text that should be truncated..." * 10},
                {"section": "Methodology", "text": "Short text"}
            ]
            formatted = generator._format_paper_sections_for_prompt(sections)
            
            assert "Introduction" in formatted
            assert "Methodology" in formatted
            # Long text should be truncated
            assert len(formatted.split("Introduction")[1].split("Methodology")[0]) < 250

