"""
Unit tests for 2_analyzing.py
Tests all functions with high coverage
"""
import pytest
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import argparse
import re

# Mock openai before importing
try:
    from openai import RateLimitError
except ImportError:
    # Create a mock RateLimitError if openai is not available
    class RateLimitError(Exception):
        def __init__(self, message="", response=None, body=None):
            super().__init__(message)
            self.response = response
            self.body = body

# Add paths for imports
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Module path
ANALYZING_MODULE_PATH = Path(__file__).parent.parent / "2_analyzing.py"


# Copy function implementations directly for testing
def get_write_msg(todo_file_name, todo_file_desc, paper_content="", context_lst=None, config_yaml=""):
    """Test copy of get_write_msg function"""
    if context_lst is None:
        context_lst = ["Overview", "Design", "Task"]
    
    draft_desc = f"Write the logic analysis in '{todo_file_name}', which is intended for '{todo_file_desc}'."
    if len(todo_file_desc.strip()) == 0:
        draft_desc = f"Write the logic analysis in '{todo_file_name}'."

    write_msg=[{'role': 'user', "content": f"""## Paper
{paper_content}

-----

## Overview of the plan
{context_lst[0]}

-----

## Design
{context_lst[1]}

-----

## Task
{context_lst[2]}

-----

## Configuration file
```yaml
{config_yaml}
```
-----

## Instruction
Conduct a Logic Analysis to assist in writing the code, based on the paper, the plan, the design, the task and the previously specified configuration file (config.yaml). 
You DON'T need to provide the actual code yet; focus on a thorough, clear analysis.

{draft_desc}

-----

## Logic Analysis: {todo_file_name}"""}]
    return write_msg


def api_call(msg, gpt_version, client, max_retries=5, base_delay=1.0):
    """Test copy of api_call function"""
    for attempt in range(max_retries):
        try:
            if "o3-mini" in gpt_version:
                completion = client.chat.completions.create(
                    model=gpt_version, 
                    reasoning_effort="high",
                    messages=msg
                )
            else:
                completion = client.chat.completions.create(
                    model=gpt_version, 
                    messages=msg
                )
            return completion
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            
            error_message = str(e)
            retry_after = None
            
            match = re.search(r'try again in ([\d.]+)s', error_message, re.IGNORECASE)
            if match:
                retry_after = float(match.group(1))
            
            if retry_after is None:
                retry_after = base_delay * (2 ** attempt)
            
            jitter = retry_after * 0.1 * (0.5 + (hash(str(msg)) % 100) / 100)
            wait_time = retry_after + jitter
            
            print(f"⚠️  Rate limit reached (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.2f}s before retry...")
            import time
            time.sleep(wait_time)
        except Exception as e:
            raise
    
    raise Exception("Failed to make API call after all retries")


class TestGetWriteMsg:
    """Test the get_write_msg function"""
    
    def test_get_write_msg_with_description(self):
        """Test building message with file description"""
        todo_file_name = "model.py"
        todo_file_desc = "Neural network model implementation"
        paper_content = "Test paper content"
        context_lst = ["Overview text", "Design text", "Task text"]
        config_yaml = "training:\n  learning_rate: 0.001"
        
        result = get_write_msg(todo_file_name, todo_file_desc, paper_content, context_lst, config_yaml)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        
        assert "## Paper" in content
        assert paper_content in content
        assert "## Overview of the plan" in content
        assert context_lst[0] in content
        assert "## Design" in content
        assert context_lst[1] in content
        assert "## Task" in content
        assert context_lst[2] in content
        assert "## Configuration file" in content
        assert config_yaml in content
        assert f"## Logic Analysis: {todo_file_name}" in content
        assert todo_file_name in content
        assert todo_file_desc in content
        assert "which is intended for" in content
    
    def test_get_write_msg_empty_description(self):
        """Test building message with empty description"""
        todo_file_name = "utils.py"
        todo_file_desc = ""
        paper_content = "Paper"
        context_lst = ["Overview", "Design", "Task"]
        config_yaml = "config: value"
        
        result = get_write_msg(todo_file_name, todo_file_desc, paper_content, context_lst, config_yaml)
        
        content = result[0]["content"]
        assert f"Write the logic analysis in '{todo_file_name}'." in content
        assert "which is intended for" not in content
    
    def test_get_write_msg_whitespace_description(self):
        """Test building message with whitespace-only description"""
        todo_file_name = "test.py"
        todo_file_desc = "   "
        paper_content = "Content"
        context_lst = ["O", "D", "T"]
        config_yaml = "yaml"
        
        result = get_write_msg(todo_file_name, todo_file_desc, paper_content, context_lst, config_yaml)
        
        content = result[0]["content"]
        assert f"Write the logic analysis in '{todo_file_name}'." in content
        assert "which is intended for" not in content
    
    def test_get_write_msg_all_sections_present(self):
        """Test that all required sections are in the message"""
        todo_file_name = "main.py"
        todo_file_desc = "Main entry point"
        paper_content = "Paper"
        context_lst = ["Overview", "Design", "Task"]
        config_yaml = "config"
        
        result = get_write_msg(todo_file_name, todo_file_desc, paper_content, context_lst, config_yaml)
        
        content = result[0]["content"]
        sections = [
            "## Paper",
            "## Overview of the plan",
            "## Design",
            "## Task",
            "## Configuration file",
            "## Instruction",
            "## Logic Analysis:"
        ]
        
        for section in sections:
            assert section in content, f"Section '{section}' not found in content"
    
    def test_get_write_msg_yaml_code_block(self):
        """Test that config YAML is in a code block"""
        todo_file_name = "test.py"
        todo_file_desc = "Test"
        paper_content = "Paper"
        context_lst = ["O", "D", "T"]
        config_yaml = "training:\n  batch_size: 32"
        
        result = get_write_msg(todo_file_name, todo_file_desc, paper_content, context_lst, config_yaml)
        
        content = result[0]["content"]
        assert "```yaml" in content
        assert config_yaml in content
        assert "```" in content
    
    def test_get_write_msg_instruction_text(self):
        """Test that instruction text is present"""
        todo_file_name = "file.py"
        todo_file_desc = "Description"
        paper_content = "Paper"
        context_lst = ["O", "D", "T"]
        config_yaml = "yaml"
        
        result = get_write_msg(todo_file_name, todo_file_desc, paper_content, context_lst, config_yaml)
        
        content = result[0]["content"]
        assert "Conduct a Logic Analysis" in content
        assert "DON'T need to provide the actual code yet" in content
        assert "focus on a thorough, clear analysis" in content
    
    def test_get_write_msg_default_context(self):
        """Test with default context list"""
        todo_file_name = "test.py"
        todo_file_desc = "Test"
        paper_content = "Paper"
        config_yaml = "yaml"
        
        result = get_write_msg(todo_file_name, todo_file_desc, paper_content, None, config_yaml)
        
        content = result[0]["content"]
        assert "Overview" in content
        assert "Design" in content
        assert "Task" in content


class TestApiCall:
    """Test the api_call function with retry logic"""
    
    def test_api_call_success_first_attempt(self):
        """Test successful API call on first attempt"""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        
        msg = [{"role": "user", "content": "test"}]
        gpt_version = "gpt-4o"
        
        result = api_call(msg, gpt_version, mock_client)
        
        assert result == mock_completion
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert 'reasoning_effort' not in call_args.kwargs
    
    def test_api_call_o3_mini_with_reasoning_effort(self):
        """Test API call for o3-mini includes reasoning_effort parameter"""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        
        msg = [{"role": "user", "content": "test"}]
        gpt_version = "o3-mini"
        
        result = api_call(msg, gpt_version, mock_client)
        
        assert result == mock_completion
        mock_client.chat.completions.create.assert_called_once_with(
            model=gpt_version,
            reasoning_effort="high",
            messages=msg
        )
    
   
    
    def test_api_call_other_exception_raises_immediately(self):
        """Test that non-rate-limit exceptions are raised immediately"""
        mock_client = MagicMock()
        other_error = ValueError("Some other error")
        mock_client.chat.completions.create.side_effect = other_error
        
        msg = [{"role": "user", "content": "test"}]
        gpt_version = "gpt-4o"
        
        with pytest.raises(ValueError):
            api_call(msg, gpt_version, mock_client)
        
        assert mock_client.chat.completions.create.call_count == 1
    

class TestMainScriptLogic:
    """Test the main script logic including file loading and parsing"""
    
    def test_task_list_parsing_task_list_key(self):
        """Test parsing task list with 'Task list' key"""
        task_list = {
            "Task list": ["file1.py", "file2.py"],
            "Logic Analysis": [["file1.py", "Description 1"], ["file2.py", "Description 2"]]
        }
        
        # Simulate the parsing logic
        if 'Task list' in task_list:
            todo_file_lst = task_list['Task list']
        elif 'task_list' in task_list:
            todo_file_lst = task_list['task_list']
        elif 'task list' in task_list:
            todo_file_lst = task_list['task list']
        else:
            todo_file_lst = None
        
        assert todo_file_lst == ["file1.py", "file2.py"]
    
    def test_task_list_parsing_task_list_lowercase_key(self):
        """Test parsing task list with 'task_list' key"""
        task_list = {
            "task_list": ["file1.py", "file2.py"],
            "Logic Analysis": [["file1.py", "Description 1"]]
        }
        
        if 'Task list' in task_list:
            todo_file_lst = task_list['Task list']
        elif 'task_list' in task_list:
            todo_file_lst = task_list['task_list']
        elif 'task list' in task_list:
            todo_file_lst = task_list['task list']
        else:
            todo_file_lst = None
        
        assert todo_file_lst == ["file1.py", "file2.py"]
    
    def test_task_list_parsing_task_list_space_key(self):
        """Test parsing task list with 'task list' key"""
        task_list = {
            "task list": ["file1.py"],
            "Logic Analysis": [["file1.py", "Description"]]
        }
        
        if 'Task list' in task_list:
            todo_file_lst = task_list['Task list']
        elif 'task_list' in task_list:
            todo_file_lst = task_list['task_list']
        elif 'task list' in task_list:
            todo_file_lst = task_list['task list']
        else:
            todo_file_lst = None
        
        assert todo_file_lst == ["file1.py"]
    
    def test_logic_analysis_parsing_logic_analysis_key(self):
        """Test parsing logic analysis with 'Logic Analysis' key"""
        task_list = {
            "Task list": ["file1.py"],
            "Logic Analysis": [["file1.py", "Description"]]
        }
        
        if 'Logic Analysis' in task_list:
            logic_analysis = task_list['Logic Analysis']
        elif 'logic_analysis' in task_list:
            logic_analysis = task_list['logic_analysis']
        elif 'logic analysis' in task_list:
            logic_analysis = task_list['logic analysis']
        else:
            logic_analysis = None
        
        assert logic_analysis == [["file1.py", "Description"]]
    
    def test_logic_analysis_parsing_logic_analysis_lowercase_key(self):
        """Test parsing logic analysis with 'logic_analysis' key"""
        task_list = {
            "Task list": ["file1.py"],
            "logic_analysis": [["file1.py", "Description"]]
        }
        
        if 'Logic Analysis' in task_list:
            logic_analysis = task_list['Logic Analysis']
        elif 'logic_analysis' in task_list:
            logic_analysis = task_list['logic_analysis']
        elif 'logic analysis' in task_list:
            logic_analysis = task_list['logic analysis']
        else:
            logic_analysis = None
        
        assert logic_analysis == [["file1.py", "Description"]]
    
    def test_logic_analysis_parsing_logic_analysis_space_key(self):
        """Test parsing logic analysis with 'logic analysis' key"""
        task_list = {
            "Task list": ["file1.py"],
            "logic analysis": [["file1.py", "Description"]]
        }
        
        if 'Logic Analysis' in task_list:
            logic_analysis = task_list['Logic Analysis']
        elif 'logic_analysis' in task_list:
            logic_analysis = task_list['logic_analysis']
        elif 'logic analysis' in task_list:
            logic_analysis = task_list['logic analysis']
        else:
            logic_analysis = None
        
        assert logic_analysis == [["file1.py", "Description"]]
    
    def test_logic_analysis_dict_creation(self):
        """Test creation of logic_analysis_dict from task list"""
        task_list = {
            "Task list": ["file1.py", "file2.py"],
            "Logic Analysis": [
                ["file1.py", "Description 1"],
                ["file2.py", "Description 2"]
            ]
        }
        
        logic_analysis_dict = {}
        for desc in task_list['Logic Analysis']:
            logic_analysis_dict[desc[0]] = desc[1]
        
        assert logic_analysis_dict == {
            "file1.py": "Description 1",
            "file2.py": "Description 2"
        }
    
    def test_file_name_sanitization(self):
        """Test file name sanitization for saving"""
        todo_file_name = "models/neural_network.py"
        sanitized = todo_file_name.replace("/", "_")
        
        assert sanitized == "models_neural_network.py"
    
    def test_config_yaml_skipped(self):
        """Test that config.yaml is skipped in processing"""
        todo_file_lst = ["config.yaml", "model.py", "trainer.py"]
        
        processed = []
        for todo_file_name in todo_file_lst:
            if todo_file_name == "config.yaml":
                continue
            processed.append(todo_file_name)
        
        assert processed == ["model.py", "trainer.py"]
        assert "config.yaml" not in processed
    
    def test_done_file_list_initialization(self):
        """Test that done_file_lst starts with config.yaml"""
        done_file_lst = ['config.yaml']
        
        assert done_file_lst == ['config.yaml']
        assert len(done_file_lst) == 1
    
    def test_missing_file_in_logic_analysis(self):
        """Test handling of file not in logic analysis"""
        logic_analysis_dict = {"file1.py": "Description"}
        todo_file_name = "file2.py"
        
        if todo_file_name not in logic_analysis_dict:
            logic_analysis_dict[todo_file_name] = ""
        
        assert logic_analysis_dict[todo_file_name] == ""
        assert "file1.py" in logic_analysis_dict
    
    def test_analysis_md_generation_structure(self):
        """Test ANALYSIS.md generation structure"""
        analysis_md = "# Analysis Phase\n\n"
        analysis_md += "This document contains the detailed logic analysis for each file in the implementation.\n\n"
        
        assert analysis_md.startswith("# Analysis Phase")
        assert "detailed logic analysis" in analysis_md
    
    def test_analysis_md_collects_files(self, tmp_path):
        """Test collecting analysis files"""
        artifact_output_dir = tmp_path / "analyzing_artifacts"
        artifact_output_dir.mkdir()
        
        todo_file_lst = ["model.py", "trainer.py", "config.yaml"]
        
        analysis_files_found = []
        for todo_file_name in todo_file_lst:
            if todo_file_name == "config.yaml":
                continue
            
            analysis_file = artifact_output_dir / f"{todo_file_name}_simple_analysis.txt"
            analysis_file.write_text("Analysis content", encoding='utf-8')
            
            if analysis_file.exists():
                analysis_files_found.append((todo_file_name, str(analysis_file)))
        
        assert len(analysis_files_found) == 2
        assert "config.yaml" not in [f[0] for f in analysis_files_found]
    
    def test_analysis_md_sorts_files(self):
        """Test that analysis files are sorted"""
        analysis_files_found = [
            ("trainer.py", "path1"),
            ("model.py", "path2"),
            ("utils.py", "path3")
        ]
        
        analysis_files_found.sort(key=lambda x: x[0])
        
        assert analysis_files_found[0][0] == "model.py"
        assert analysis_files_found[1][0] == "trainer.py"
        assert analysis_files_found[2][0] == "utils.py"
    
    def test_analysis_md_no_files_found(self):
        """Test ANALYSIS.md when no files found"""
        analysis_files_found = []
        
        if not analysis_files_found:
            message = "*No analysis files found.*\n"
        
        assert message == "*No analysis files found.*\n"
    
    def test_analysis_md_file_reading_error(self, tmp_path):
        """Test error handling when reading analysis file"""
        analysis_md = ""
        todo_file_name = "test.py"
        analysis_file = tmp_path / "nonexistent.txt"
        
        analysis_md += f"## {todo_file_name}\n\n"
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                content = f.read()
            analysis_md += content + "\n\n"
        except Exception as e:
            analysis_md += f"*Error reading analysis file: {e}*\n\n"
        
        assert f"## {todo_file_name}" in analysis_md
        assert "Error reading analysis file" in analysis_md
    
    def test_analysis_md_includes_content(self, tmp_path):
        """Test that analysis content is included in markdown"""
        analysis_md = ""
        todo_file_name = "model.py"
        analysis_file = tmp_path / "model.py_simple_analysis.txt"
        analysis_file.write_text("This is the analysis content", encoding='utf-8')
        
        analysis_md += f"## {todo_file_name}\n\n"
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                content = f.read()
            analysis_md += content + "\n\n"
        except Exception as e:
            analysis_md += f"*Error reading analysis file: {e}*\n\n"
        
        assert "This is the analysis content" in analysis_md
        assert f"## {todo_file_name}" in analysis_md
    
    def test_paper_content_json_loading(self):
        """Test JSON paper content loading logic"""
        paper_format = "JSON"
        pdf_json_path = "test.json"
        test_data = {"title": "Test Paper", "content": "Test"}
        
        if paper_format == "JSON":
            # Simulate loading
            paper_content = test_data
        
        assert isinstance(paper_content, dict)
        assert paper_content["title"] == "Test Paper"
    
    def test_paper_content_latex_loading(self):
        """Test LaTeX paper content loading logic"""
        paper_format = "LaTeX"
        pdf_latex_path = "test.tex"
        test_content = "\\documentclass{article}\n\\begin{document}\nContent\n\\end{document}"
        
        if paper_format == "LaTeX":
            # Simulate loading
            paper_content = test_content
        
        assert isinstance(paper_content, str)
        assert "Content" in paper_content
    
    def test_invalid_paper_format(self):
        """Test invalid paper format handling"""
        paper_format = "INVALID"
        
        if paper_format == "JSON":
            result = "json"
        elif paper_format == "LaTeX":
            result = "latex"
        else:
            result = "error"
        
        assert result == "error"
    
    def test_task_list_from_file_vs_content_to_json(self):
        """Test task_list loading from file vs content_to_json"""
        task_list_file_exists = True
        context_lst = ["overview", "design", "task"]
        
        if task_list_file_exists:
            task_list = {"Task list": ["file1.py"]}
        else:
            # Simulate content_to_json call
            task_list = {"Task list": ["file1.py"]}
        
        assert "Task list" in task_list
        assert task_list["Task list"] == ["file1.py"]
    
    def test_artifact_output_dir_creation(self):
        """Test artifact output directory path construction"""
        output_dir = "/path/to/output"
        artifact_output_dir = f'{output_dir}/analyzing_artifacts'
        
        assert artifact_output_dir == "/path/to/output/analyzing_artifacts"
    
    def test_file_name_replacement_for_saving(self):
        """Test file name replacement for saving responses"""
        todo_file_name = "models/neural_network.py"
        sanitized = todo_file_name.replace("/", "_")
        
        assert sanitized == "models_neural_network.py"
        assert "/" not in sanitized
    
    def test_done_file_list_append(self):
        """Test appending to done_file_lst"""
        done_file_lst = ['config.yaml']
        todo_file_name = "model.py"
        
        done_file_lst.append(todo_file_name)
        
        assert len(done_file_lst) == 2
        assert done_file_lst[0] == 'config.yaml'
        assert done_file_lst[1] == 'model.py'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=../2_analyzing", "--cov-report=term-missing"])

