import json
from pathlib import Path
from unittest.mock import patch
import pytest

from parser import codegen_prep, ep2c_parse


def test_codegen_prep_basic(tmp_path):
    """
    Test that codegen_prep correctly loads a JSON content list,
    concatenates text, and stores images as separate entries.
    """

    # Simulate input document paths.
    doc_path = tmp_path / "doc1.pdf"
    doc_path.touch()  # create empty file
    doc_paths = [doc_path]

    # Simulate MinerU default directory structure.
    output_dir = tmp_path
    auto_dir = output_dir / "doc1" / "auto"
    auto_dir.mkdir(parents=True)

    fake_json = [
        {"type": "text", "text": "Hello "},
        {"type": "text", "text": "world!"},
        {"type": "image", "img_path": "img001.png", "image_caption": "caption"}
    ]

    json_path = auto_dir / "doc1_content_list.json"
    json_path.write_text(json.dumps(fake_json))

    result = codegen_prep(doc_paths, output_dir)

    assert len(result) == 1
    assert result[0]["document"] == "doc1"
    assert len(result[0]["content"]) == 2

    text_entry = result[0]["content"][0]
    image_entry = result[0]["content"][1]

    assert text_entry["type"] == "text"
    assert "Hello " in text_entry["text"]
    assert "world!" in text_entry["text"]

    assert image_entry["type"] == "image"
    assert isinstance(image_entry["path"], Path)
    assert image_entry["path"].name == "img001.png"

def test_codegen_prep_missing_json(tmp_path, capsys):
    doc_path = tmp_path / "docA.pdf"
    doc_path.touch()
    doc_paths = [doc_path]

    output_dir = tmp_path
    auto_dir = output_dir / "docA" / "auto"
    auto_dir.mkdir(parents=True)

    # Missing JSON file should print error but not crash.
    codegen_prep(doc_paths, output_dir)

    captured = capsys.readouterr()
    assert "not found" in captured.err.lower()

@patch("your_module._parse_doc")
def test_ep2c_parse_valid(mock_parse):
    docs = [
        ("file1.pdf", "en"),
        (Path("file2.pdf"), "ch")
    ]

    ep2c_parse(docs, "/tmp/output")

    assert mock_parse.called

    _, kwargs = mock_parse.call_args
    assert "path_list" in kwargs
    assert kwargs["langs"] == ["en", "ch"]
    assert kwargs["output_dir"] == "/tmp/output"

def test_ep2c_parse_invalid_docs_type(capsys):
    with pytest.raises(SystemExit):
        ep2c_parse("not_a_list", "/tmp/out")

    assert "documents must be in a list" in capsys.readouterr().err.lower()

def test_ep2c_parse_invalid_output_type(capsys):
    with pytest.raises(SystemExit):
        ep2c_parse([], 123)  # not str or Path

    assert "output path must be a string or path" in capsys.readouterr().err.lower()

def test_ep2c_parse_invalid_tuple_structure(capsys):
    docs = [
        ("file.pdf",),  # bad: only 1 element
    ]

    with pytest.raises(SystemExit):
        ep2c_parse(docs, "/tmp/out")

    assert "document list should be a list" in capsys.readouterr().err.lower()

def test_ep2c_parse_unsupported_language(capsys):
    docs = [("file.pdf", "german")]

    with pytest.raises(SystemExit):
        ep2c_parse(docs, "/tmp/out")

    assert "language option not supported" in capsys.readouterr().err.lower()
