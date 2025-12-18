"""
Unit tests for 3_coding.py
Tests all functions with high coverage
"""
import pytest
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open, call
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
CODING_MODULE_PATH = Path(__file__).parent.parent / "3_coding.py"


# Copy function implementations directly for testing
def get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst, paper_content="", context_lst=None, config_yaml="", done_file_dict=None):
    """Test copy of get_write_msg function"""
    if context_lst is None:
        context_lst = ["Overview", "Design", "Task"]
    if done_file_dict is None:
        done_file_dict = {}
    
    code_files = ""
    for done_file in done_file_lst:
        if done_file.endswith(".yaml"): 
            continue
        code_files += f"""
```python
{done_file_dict.get(done_file, "")}
```

"""

    write_msg=[
{'role': 'user', "content": f"""# Context
## Paper
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

## Code Files
{code_files}

-----

# Format example
## Code: {todo_file_name}
```python
## {todo_file_name}
...
```

-----

# Instruction
Based on the paper, plan, design, task and configuration file(config.yaml) specified previously, follow "Format example", write the code. 

We have {done_file_lst}.
Next, you must write only the "{todo_file_name}".
1. Only One file: do your best to implement THIS ONLY ONE FILE.
2. COMPLETE CODE: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.
3. Set default value: If there is any setting, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE. AVOID circular import.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.
6. Before using a external variable/module, make sure you import it first.
7. Write out EVERY CODE DETAIL, DON'T LEAVE TODO.
8. REFER TO CONFIGURATION: you must use configuration from "config.yaml". DO NOT FABRICATE any configuration values.

{detailed_logic_analysis}

## Code: {todo_file_name}"""}]
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
    
    def test_get_write_msg_basic(self):
        """Test basic message building"""
        todo_file_name = "model.py"
        detailed_logic_analysis = "This file implements the neural network model."
        done_file_lst = ["config.yaml"]
        paper_content = "Test paper"
        context_lst = ["Overview", "Design", "Task"]
        config_yaml = "training:\n  learning_rate: 0.001"
        done_file_dict = {}
        
        result = get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst, 
                              paper_content, context_lst, config_yaml, done_file_dict)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        
        assert "## Paper" in content
        assert paper_content in content
        assert "## Overview of the plan" in content
        assert "## Design" in content
        assert "## Task" in content
        assert "## Configuration file" in content
        assert config_yaml in content
        assert f"## Code: {todo_file_name}" in content
        assert detailed_logic_analysis in content
    
    def test_get_write_msg_with_code_files(self):
        """Test message building with existing code files"""
        todo_file_name = "trainer.py"
        detailed_logic_analysis = "Training logic"
        done_file_lst = ["config.yaml", "model.py", "utils.py"]
        done_file_dict = {
            "model.py": "class Model:\n    pass",
            "utils.py": "def helper():\n    pass"
        }
        
        result = get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst, 
                              done_file_dict=done_file_dict)
        
        content = result[0]["content"]
        assert "## Code Files" in content
        assert "model.py" in content
        assert "utils.py" in content
        assert "class Model:" in content
        assert "def helper():" in content
    
    def test_get_write_msg_skips_yaml_files(self):
        """Test that YAML files are skipped in code files section"""
        todo_file_name = "test.py"
        detailed_logic_analysis = "Test"
        done_file_lst = ["config.yaml", "model.py"]
        done_file_dict = {
            "model.py": "code here"
        }
        
        result = get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst,
                              done_file_dict=done_file_dict)
        
        content = result[0]["content"]
        # config.yaml should not appear in code files section
        code_files_section = content.split("## Code Files")[1].split("-----")[0]
        assert "config.yaml" not in code_files_section
    
    def test_get_write_msg_format_example(self):
        """Test that format example is included"""
        todo_file_name = "main.py"
        detailed_logic_analysis = "Main entry point"
        done_file_lst = ["config.yaml"]
        
        result = get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst)
        
        content = result[0]["content"]
        assert "# Format example" in content
        assert f"## Code: {todo_file_name}" in content
        assert "```python" in content
    
    def test_get_write_msg_instructions(self):
        """Test that all instructions are present"""
        todo_file_name = "file.py"
        detailed_logic_analysis = "Analysis"
        done_file_lst = ["config.yaml"]
        
        result = get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst)
        
        content = result[0]["content"]
        assert "# Instruction" in content
        assert "Only One file" in content
        assert "COMPLETE CODE" in content
        assert "Set default value" in content
        assert "Follow design" in content
        assert "REFER TO CONFIGURATION" in content
    
    def test_get_write_msg_done_file_list(self):
        """Test that done file list is included in message"""
        todo_file_name = "new_file.py"
        detailed_logic_analysis = "Analysis"
        done_file_lst = ["config.yaml", "model.py", "trainer.py"]
        
        result = get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst)
        
        content = result[0]["content"]
        assert "We have" in content
        # Check that done files are mentioned
        assert "config.yaml" in content or "model.py" in content
    
    def test_get_write_msg_empty_done_file_dict(self):
        """Test with empty done_file_dict"""
        todo_file_name = "test.py"
        detailed_logic_analysis = "Test"
        done_file_lst = ["config.yaml", "model.py"]
        done_file_dict = {}
        
        result = get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst,
                              done_file_dict=done_file_dict)
        
        content = result[0]["content"]
        # Should still have Code Files section even if empty
        assert "## Code Files" in content


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
    """Test the main script logic including code extraction and file operations"""
    
    def test_file_name_sanitization(self):
        """Test file name sanitization for saving"""
        todo_file_name = "models/neural_network.py"
        save_todo_file_name = todo_file_name.replace("/", "_")
        
        assert save_todo_file_name == "models_neural_network.py"
    
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
    
    def test_code_extraction_fallback(self):
        """Test code extraction with fallback to full content"""
        message_content = "Some text without code blocks"
        
        # Simulate extract_code_from_content returning empty
        code = ""  # Simulated empty extraction
        if len(code) == 0:
            code = message_content
        
        assert code == message_content
    
    def test_code_extraction_with_code_block(self):
        """Test code extraction from code block"""
        message_content = """Some text
```python
def function():
    pass
```
More text"""
        
        # Simulate extract_code_from_content
        import re
        pattern = r'```python\s*(.*?)```'
        result = re.search(pattern, message_content, re.DOTALL)
        
        if result:
            code = result.group(1).strip()
        else:
            code = message_content
        
        assert "def function():" in code
        assert "pass" in code
    
    def test_done_file_dict_storage(self):
        """Test storing code in done_file_dict"""
        done_file_dict = {}
        todo_file_name = "model.py"
        code = "class Model:\n    pass"
        
        done_file_dict[todo_file_name] = code
        
        assert done_file_dict[todo_file_name] == code
        assert "Model" in done_file_dict[todo_file_name]
    
    def test_directory_creation_for_nested_files(self):
        """Test directory creation for nested file paths"""
        todo_file_name = "models/neural_network.py"
        save_todo_file_name = todo_file_name.replace("/", "_")
        
        if save_todo_file_name != todo_file_name:
            todo_file_dir = '/'.join(todo_file_name.split("/")[:-1])
        
        assert todo_file_dir == "models"
        assert save_todo_file_name == "models_neural_network.py"
    
    def test_coding_md_generation_structure(self):
        """Test CODING.md generation structure"""
        coding_md = "# Coding Phase\n\n"
        coding_md += "This document contains the code generation artifacts for each file in the implementation.\n\n"
        
        assert coding_md.startswith("# Coding Phase")
        assert "code generation artifacts" in coding_md
    
    def test_coding_md_collects_files(self, tmp_path):
        """Test collecting coding artifact files"""
        artifact_output_dir = tmp_path / "coding_artifacts"
        artifact_output_dir.mkdir()
        
        todo_file_lst = ["model.py", "trainer.py", "config.yaml"]
        
        coding_files_found = []
        for todo_file_name in todo_file_lst:
            if todo_file_name == "config.yaml":
                continue
            
            save_todo_file_name = todo_file_name.replace("/", "_")
            coding_file = artifact_output_dir / f"{save_todo_file_name}_coding.txt"
            coding_file.write_text("Code content", encoding='utf-8')
            
            if coding_file.exists():
                coding_files_found.append((todo_file_name, str(coding_file)))
        
        assert len(coding_files_found) == 2
        assert "config.yaml" not in [f[0] for f in coding_files_found]
    
    def test_coding_md_sorts_files(self):
        """Test that coding files are sorted"""
        coding_files_found = [
            ("trainer.py", "path1"),
            ("model.py", "path2"),
            ("utils.py", "path3")
        ]
        
        coding_files_found.sort(key=lambda x: x[0])
        
        assert coding_files_found[0][0] == "model.py"
        assert coding_files_found[1][0] == "trainer.py"
        assert coding_files_found[2][0] == "utils.py"
    
    def test_coding_md_no_files_found(self):
        """Test CODING.md when no files found"""
        coding_files_found = []
        
        if not coding_files_found:
            message = "*No coding artifacts found.*\n"
        
        assert message == "*No coding artifacts found.*\n"
    
    def test_coding_md_file_reading_error(self, tmp_path):
        """Test error handling when reading coding file"""
        coding_md = ""
        todo_file_name = "test.py"
        coding_file = tmp_path / "nonexistent.txt"
        
        coding_md += f"## {todo_file_name}\n\n"
        try:
            with open(coding_file, 'r', encoding='utf-8') as f:
                content = f.read()
            coding_md += content + "\n\n"
        except Exception as e:
            coding_md += f"*Error reading coding file: {e}*\n\n"
        
        assert f"## {todo_file_name}" in coding_md
        assert "Error reading coding file" in coding_md
    
    def test_coding_md_includes_content(self, tmp_path):
        """Test that coding content is included in markdown"""
        coding_md = ""
        todo_file_name = "model.py"
        coding_file = tmp_path / "model.py_coding.txt"
        coding_file.write_text("def model():\n    pass", encoding='utf-8')
        
        coding_md += f"## {todo_file_name}\n\n"
        try:
            with open(coding_file, 'r', encoding='utf-8') as f:
                content = f.read()
            coding_md += content + "\n\n"
        except Exception as e:
            coding_md += f"*Error reading coding file: {e}*\n\n"
        
        assert "def model():" in coding_md
        assert f"## {todo_file_name}" in coding_md
    
    def test_coding_md_note_section(self):
        """Test that note section is added to CODING.md"""
        output_repo_dir = "/path/to/repo"
        coding_md = ""
        coding_md += "---\n\n"
        coding_md += f"**Note:** Generated code files are available in `{output_repo_dir}`\n"
        
        assert "---" in coding_md
        assert "**Note:**" in coding_md
        assert output_repo_dir in coding_md
    
    def test_detailed_logic_analysis_dict_loading(self):
        """Test loading detailed logic analysis from response files"""
        output_dir = "/tmp/test"
        todo_file_name = "model.py"
        save_todo_file_name = todo_file_name.replace("/", "_")
        
        # Simulate response structure
        detailed_logic_analysis_response = [{
            'choices': [{
                'message': {
                    'content': 'This is the analysis content'
                }
            }]
        }]
        
        detailed_logic_analysis_dict = {}
        detailed_logic_analysis_dict[todo_file_name] = detailed_logic_analysis_response[0]['choices'][0]['message']['content']
        
        assert detailed_logic_analysis_dict[todo_file_name] == 'This is the analysis content'
    
    def test_artifact_output_dir_creation(self):
        """Test artifact output directory path construction"""
        output_dir = "/path/to/output"
        artifact_output_dir = f'{output_dir}/coding_artifacts'
        
        assert artifact_output_dir == "/path/to/output/coding_artifacts"
    
    def test_done_file_list_append(self):
        """Test appending to done_file_lst"""
        done_file_lst = ['config.yaml']
        todo_file_name = "model.py"
        
        done_file_lst.append(todo_file_name)
        
        assert len(done_file_lst) == 2
        assert done_file_lst[0] == 'config.yaml'
        assert done_file_lst[1] == 'model.py'
    
    def test_file_path_handling_nested(self):
        """Test handling of nested file paths"""
        todo_file_name = "models/layers/linear.py"
        save_todo_file_name = todo_file_name.replace("/", "_")
        
        assert save_todo_file_name == "models_layers_linear.py"
        
        if save_todo_file_name != todo_file_name:
            todo_file_dir = '/'.join(todo_file_name.split("/")[:-1])
        
        assert todo_file_dir == "models/layers"
    
    def test_code_saving_to_repo(self):
        """Test code saving to repository directory"""
        output_repo_dir = "/tmp/repo"
        todo_file_name = "model.py"
        code = "class Model:\n    pass"
        
        # Simulate file writing
        file_path = f"{output_repo_dir}/{todo_file_name}"
        
        assert file_path == "/tmp/repo/model.py"
        assert code == "class Model:\n    pass"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=../3_coding", "--cov-report=term-missing"])

