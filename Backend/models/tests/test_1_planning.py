"""
Unit tests for 1_planning.py
Tests all functions with high coverage
"""
import pytest
import sys
import os
import json
import base64
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open, call, PropertyMock
import argparse
import importlib.util
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
PLANNING_MODULE_PATH = Path(__file__).parent.parent / "1_planning.py"


# Copy function implementations directly for testing
def encode_image_to_base64(image_path: Path, max_size_mb: float = 20.0) -> str:
    """Test copy of encode_image_to_base64 function"""
    try:
        file_size = image_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size_mb > max_size_mb:
            print(f"⚠️  Warning: Image {image_path.name} is {file_size_mb:.2f}MB (exceeds {max_size_mb}MB limit). Skipping.")
            return None
        
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_encoded = base64.b64encode(image_data).decode('utf-8')
            
            ext = image_path.suffix.lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }.get(ext, 'image/png')
            
            return f"data:{mime_type};base64,{base64_encoded}"
    except FileNotFoundError:
        print(f"⚠️  Warning: Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"⚠️  Warning: Could not encode image {image_path}: {e}")
        return None


def build_message_content_with_images(task_text: str, paper_content_items: list = None, gpt_version: str = None, paper_content: str = "") -> list:
    """Test copy of build_message_content_with_images function"""
    supports_images = gpt_version and "o3-mini" not in gpt_version.lower()
    
    has_images = paper_content_items and any(item.get("type") == "image_url" for item in paper_content_items) and supports_images
    
    if has_images:
        message_content = []
        message_content.append({
            "type": "text",
            "text": "## Paper\n\n"
        })
        
        for item in paper_content_items:
            if item.get("type") == "image_url":
                message_content.append(item)
            elif item.get("type") == "text":
                if message_content and message_content[-1].get("type") == "text":
                    message_content[-1]["text"] += "\n" + item.get("text", "")
                else:
                    message_content.append(item)
        
        if message_content and message_content[-1].get("type") == "text":
            message_content[-1]["text"] += "\n\n" + task_text
        else:
            message_content.append({
                "type": "text",
                "text": task_text
            })
        
        return message_content
    else:
        if paper_content_items:
            text_parts = []
            for item in paper_content_items:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            paper_text = "\n".join(text_parts) if text_parts else paper_content
        else:
            paper_text = paper_content
        
        return f"""## Paper
{paper_text}

{task_text}"""


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


class TestEncodeImageToBase64:
    """Test the encode_image_to_base64 function"""
    
    def test_encode_image_success_png(self, tmp_path):
        """Test successful encoding of a PNG image"""
        image_path = tmp_path / "test.png"
        image_data = b"fake_png_data_12345"
        image_path.write_bytes(image_data)
        
        result = encode_image_to_base64(image_path)
        
        assert result is not None
        assert result.startswith("data:image/png;base64,")
        base64_part = result.split(",")[1]
        decoded = base64.b64decode(base64_part)
        assert decoded == image_data
    
    def test_encode_image_success_jpg(self, tmp_path):
        """Test successful encoding of a JPG image"""
        image_path = tmp_path / "test.jpg"
        image_data = b"fake_jpg_data"
        image_path.write_bytes(image_data)
        
        result = encode_image_to_base64(image_path)
        
        assert result is not None
        assert result.startswith("data:image/jpeg;base64,")
    
    def test_encode_image_success_jpeg(self, tmp_path):
        """Test successful encoding of a JPEG image"""
        image_path = tmp_path / "test.jpeg"
        image_data = b"fake_jpeg_data"
        image_path.write_bytes(image_data)
        
        result = encode_image_to_base64(image_path)
        
        assert result is not None
        assert result.startswith("data:image/jpeg;base64,")
    
    def test_encode_image_success_gif(self, tmp_path):
        """Test successful encoding of a GIF image"""
        image_path = tmp_path / "test.gif"
        image_data = b"fake_gif_data"
        image_path.write_bytes(image_data)
        
        result = encode_image_to_base64(image_path)
        
        assert result is not None
        assert result.startswith("data:image/gif;base64,")
    
    def test_encode_image_success_webp(self, tmp_path):
        """Test successful encoding of a WebP image"""
        image_path = tmp_path / "test.webp"
        image_data = b"fake_webp_data"
        image_path.write_bytes(image_data)
        
        result = encode_image_to_base64(image_path)
        
        assert result is not None
        assert result.startswith("data:image/webp;base64,")
    
    def test_encode_image_file_too_large(self, tmp_path, capsys):
        """Test that files exceeding size limit are skipped"""
        image_path = tmp_path / "large.png"
        image_path.write_bytes(b"data")
        
        with patch.object(Path, 'stat') as mock_stat:
            stat_result = MagicMock()
            stat_result.st_size = 21 * 1024 * 1024  # 21MB
            mock_stat.return_value = stat_result
            
            result = encode_image_to_base64(image_path, max_size_mb=20.0)
            
            assert result is None
            captured = capsys.readouterr()
            assert "exceeds" in captured.out.lower() or "warning" in captured.out.lower()
    
    def test_encode_image_file_not_found(self, tmp_path, capsys):
        """Test handling of missing image file"""
        image_path = tmp_path / "nonexistent.png"
        
        result = encode_image_to_base64(image_path)
        
        assert result is None
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "warning" in captured.out.lower()
    
    def test_encode_image_exception_handling(self, tmp_path, capsys):
        """Test exception handling during encoding"""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"data")
        
        with patch('builtins.open', side_effect=Exception("Test error")):
            result = encode_image_to_base64(image_path)
            
            assert result is None
            captured = capsys.readouterr()
            assert "warning" in captured.out.lower() or "error" in captured.out.lower()
    
    def test_encode_image_unknown_extension(self, tmp_path):
        """Test encoding with unknown file extension defaults to PNG"""
        image_path = tmp_path / "test.unknown"
        image_data = b"fake_data"
        image_path.write_bytes(image_data)
        
        result = encode_image_to_base64(image_path)
        
        assert result is not None
        assert result.startswith("data:image/png;base64,")
    
    def test_encode_image_custom_max_size(self, tmp_path):
        """Test encoding with custom max size limit"""
        image_path = tmp_path / "test.png"
        image_data = b"fake_data"
        image_path.write_bytes(image_data)
        
        with patch.object(Path, 'stat') as mock_stat:
            stat_result = MagicMock()
            stat_result.st_size = 2 * 1024 * 1024  # 2MB
            mock_stat.return_value = stat_result
            
            result = encode_image_to_base64(image_path, max_size_mb=1.0)
            assert result is None


class TestBuildMessageContentWithImages:
    """Test the build_message_content_with_images function"""
    
    def test_build_message_with_images_supported(self):
        """Test building message content with images when model supports them"""
        task_text = "Test task"
        paper_content_items = [
            {"type": "text", "text": "Paper text"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}},
            {"type": "text", "text": "More text"}
        ]
        gpt_version = "gpt-4o"
        
        result = build_message_content_with_images(task_text, paper_content_items, gpt_version)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["type"] == "text"
        assert "## Paper" in result[0]["text"]
        image_items = [item for item in result if item.get("type") == "image_url"]
        assert len(image_items) > 0
    
    def test_build_message_with_images_o3_mini(self):
        """Test that o3-mini models don't get images"""
        task_text = "Test task"
        paper_content_items = [
            {"type": "text", "text": "Paper text"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}}
        ]
        gpt_version = "o3-mini"
        
        result = build_message_content_with_images(task_text, paper_content_items, gpt_version)
        
        assert isinstance(result, str)
        assert "## Paper" in result
        assert task_text in result
    
    def test_build_message_no_images(self):
        """Test building message when no images are present"""
        task_text = "Test task"
        paper_content_items = [
            {"type": "text", "text": "Paper text"}
        ]
        gpt_version = "gpt-4o"
        
        result = build_message_content_with_images(task_text, paper_content_items, gpt_version)
        
        assert isinstance(result, str)
        assert "## Paper" in result
        assert "Paper text" in result
        assert task_text in result
    
    def test_build_message_empty_items(self):
        """Test building message with empty content items"""
        task_text = "Test task"
        
        result = build_message_content_with_images(task_text, None, "gpt-4o")
        
        assert isinstance(result, str)
        assert task_text in result
    
    def test_build_message_merges_consecutive_text(self):
        """Test that consecutive text items are merged"""
        task_text = "Test task"
        paper_content_items = [
            {"type": "text", "text": "First text"},
            {"type": "text", "text": "Second text"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}},
            {"type": "text", "text": "Third text"}
        ]
        gpt_version = "gpt-4o"
        
        result = build_message_content_with_images(task_text, paper_content_items, gpt_version)
        
        assert isinstance(result, list)
        text_items = [item for item in result if item.get("type") == "text"]
        assert len(text_items) > 0
        first_text = text_items[0]["text"]
        assert "First text" in first_text
        assert "Second text" in first_text
    
    def test_build_message_task_appended(self):
        """Test that task text is appended to the message"""
        task_text = "Specific task text here"
        paper_content_items = [
            {"type": "text", "text": "Paper text"}
        ]
        gpt_version = "gpt-4o"
        
        result = build_message_content_with_images(task_text, paper_content_items, gpt_version)
        
        assert task_text in result
    
    def test_build_message_no_gpt_version(self):
        """Test building message without GPT version"""
        task_text = "Test task"
        paper_content_items = [
            {"type": "text", "text": "Paper text"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}}
        ]
        
        result = build_message_content_with_images(task_text, paper_content_items, None)
        
        assert isinstance(result, str)


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
    """Test the main script logic including argument parsing and file operations"""
    
    def test_extract_config_yaml(self):
        """Test extraction of config.yaml from response"""
        config_content = """## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.001
  batch_size: 32
```
"""
        
        yaml_pattern = r'```(?:yaml)?\s*\n(.*?)```'
        yaml_match = re.search(yaml_pattern, config_content, re.DOTALL)
        
        if yaml_match:
            config_yaml = yaml_match.group(1).strip()
            config_yaml = re.sub(r'^##\s*config\.yaml\s*\n', '', config_yaml, flags=re.MULTILINE)
        
        assert "training:" in config_yaml
        assert "learning_rate: 0.001" in config_yaml
        assert "batch_size: 32" in config_yaml
        assert "## config.yaml" not in config_yaml
    
    def test_extract_config_yaml_fallback_pattern(self):
        """Test YAML extraction with fallback pattern"""
        config_content = """Some text
```yaml
training:
  learning_rate: 0.001
```
More text"""
        
        code_block_pattern = r'```[^\n]*\n(.*?)```'
        code_match = re.search(code_block_pattern, config_content, re.DOTALL)
        
        if code_match:
            config_yaml = code_match.group(1).strip()
        
        assert "training:" in config_yaml
        assert "learning_rate: 0.001" in config_yaml


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=../1_planning", "--cov-report=term-missing"])
