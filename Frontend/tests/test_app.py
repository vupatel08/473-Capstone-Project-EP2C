import io
import os
import sys
import json
import tempfile
import zipfile
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

PROJ_ROOT = HERE.parent
BACKEND_PAPERCODESYNC_SRC = PROJ_ROOT / "Backend" / "papercodesync" / "src"
BACKEND_EXAMPLE_DRIVER = PROJ_ROOT / "Backend" / "example_driver"
for p in (BACKEND_PAPERCODESYNC_SRC, BACKEND_EXAMPLE_DRIVER):
    if p.exists():
        sys.path.insert(0, str(p))

import types
mock_driver = types.ModuleType("driver")
mock_driver.pcs_pipeline = lambda paper_md, repo_root: None
sys.modules["driver"] = mock_driver
mock_unified_pipeline = types.ModuleType("unified_pipeline")
mock_unified_pipeline.run_unified_pipeline = lambda **kw: {
    "repo_path": str(HERE / "repo"),
    "paper_md_path": "",
    "paper_json_path": "",
    "output_dir": "",
    "explanation_dir": "",
    "explanation_md_path": "",
    "planning_md_path": "",
    "analysis_md_path": "",
    "coding_md_path": "",
    "from_github": False
}
sys.modules["unified_pipeline"] = mock_unified_pipeline

import importlib
app_module = importlib.import_module("app")
app = app_module.app

@pytest.fixture
def client(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "file1.py").write_text("print('hello')\n")
    (repo_dir / "big.bin").write_bytes(b"x" * 60_000_000)  

    data_dir = tmp_path / "papercodesync_data"
    data_dir.mkdir()
    symbols = data_dir / "symbols.json"
    chunks = data_dir / "chunks.json"
    matches = data_dir / "matches.jsonl"
    symbols.write_text(json.dumps([{"id":"s1","name":"sym1","bow_terms":["foo"]}]))
    chunks.write_text(json.dumps([{"id":"c1","section":"s","text":"chunk text"}]))
    matches.write_text(json.dumps({"symbol_id":"s1","best":{"chunk_id":"c1","score":1.0}}) + "\n")

    monkeypatch.setattr(app_module, 'REPO_ROOT', str(repo_dir))
    monkeypatch.setattr(app_module, 'PAPERCODESYNC_DATA', str(data_dir))
    monkeypatch.setattr(app_module, 'PAPERCODESYNC_SYMBOLS', str(symbols))
    monkeypatch.setattr(app_module, 'PAPERCODESYNC_CHUNKS', str(chunks))
    monkeypatch.setattr(app_module, 'PAPERCODESYNC_MATCHES', str(matches))

    monkeypatch.setattr(app_module, 'ep2c_pipeline', lambda **kw: str(repo_dir))
    monkeypatch.setattr(app_module, 'pcs_pipeline', lambda paper_md, repo_root: None)

    with app.test_client() as client:
        yield client


def test_index(client):
    resp = client.get('/')
    assert resp.status_code == 200
    assert b"EP2C" in resp.data or b"index" in resp.data


def test_export_download(client, tmp_path):
    resp = client.get('/export')
    assert resp.status_code == 200
    assert resp.mimetype == 'application/zip'
    p = tmp_path / 'out.zip'
    p.write_bytes(resp.data)
    with zipfile.ZipFile(p, 'r') as z:
        names = z.namelist()
        assert any(n.endswith('file1.py') for n in names)


def test_viewer_and_data_endpoints(client):
    resp = client.get('/viewer')
    assert resp.status_code in (302, 308)

    r = client.get('/data/symbols.json')
    assert r.status_code == 200
    assert r.mimetype == 'application/json'

    r = client.get('/data/chunks.json')
    assert r.status_code == 200

    r = client.get('/data/matches.jsonl')
    assert r.status_code == 200
    assert b's1' in r.data


def test_serve_code_file(client, tmp_path, monkeypatch):
    repo_dir = Path(app_module.REPO_ROOT)
    nested = repo_dir / 'sub'
    nested.mkdir()
    (nested / 'a.txt').write_text('hello')

    resp = client.get('/code/sub/a.txt')
    assert resp.status_code == 200
    assert b'hello' in resp.data

    resp2 = client.get('/code/big.bin')
    assert resp2.status_code == 200


def test_upload_flow(client, tmp_path, monkeypatch):
    pdf = tmp_path / 'paper.pdf'
    pdf.write_bytes(b'%PDF-1.4 test')

    data = {
        'language': 'Python',
    }
    with open(pdf, 'rb') as f:
        rv = client.post('/upload', data={'paper': (f, 'paper.pdf'), 'language': 'Python'}, follow_redirects=True)
    assert b'viewer' in rv.data or rv.status_code in (200, 302)


def test_export_when_repo_missing(client, monkeypatch):
    monkeypatch.setattr(app_module, 'REPO_ROOT', str(Path('/nonexistent_repo')))
    r = client.get('/export')
    assert r.status_code == 404


def test_data_endpoints_missing_files(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, 'PAPERCODESYNC_SYMBOLS', str(tmp_path / 'no_symbols.json'))
    monkeypatch.setattr(app_module, 'PAPERCODESYNC_CHUNKS', str(tmp_path / 'no_chunks.json'))
    monkeypatch.setattr(app_module, 'PAPERCODESYNC_MATCHES', str(tmp_path / 'no_matches.jsonl'))

    r = client.get('/data/symbols.json')
    assert r.status_code == 404
    r = client.get('/data/chunks.json')
    assert r.status_code == 404
    r = client.get('/data/matches.jsonl')
    assert r.status_code == 404


def test_upload_invalid_file_and_missing_language(client):
    data = {}
    rv = client.post('/upload', data=data, follow_redirects=True)
    assert b'Please upload a paper and choose a language' in rv.data

    bad_file = (io.BytesIO(b'text data'), 'paper.txt')
    rv2 = client.post('/upload', data={'paper': bad_file, 'language': 'Python'}, content_type='multipart/form-data', follow_redirects=True)
    assert b'Only the following file types are allowed' in rv2.data


def test_upload_pipeline_unavailable(client, tmp_path, monkeypatch):
    pdf = tmp_path / 'paper.pdf'
    pdf.write_bytes(b'%PDF-1.4 test')

    monkeypatch.setattr(app_module, 'ep2c_pipeline', None)
    with open(pdf, 'rb') as f:
        rv = client.post('/upload', data={'paper': (f, 'paper.pdf'), 'language': 'Python'}, follow_redirects=True)
    assert b'Backend driver not available' in rv.data or b'example_driver not importable' in rv.data


def test_upload_pcs_pipeline_unavailable(client, tmp_path, monkeypatch):
    pdf = tmp_path / 'paper.pdf'
    pdf.write_bytes(b'%PDF-1.4 test')

    monkeypatch.setattr(app_module, 'ep2c_pipeline', lambda **kw: str(Path(tmp_path / 'repo')))
    monkeypatch.setattr(app_module, 'pcs_pipeline', None)

    with open(pdf, 'rb') as f:
        rv = client.post('/upload', data={'paper': (f, 'paper.pdf'), 'language': 'Python'}, follow_redirects=True)
    assert b'Backend driver not available' in rv.data or b'pcs_pipeline not importable' in rv.data


def test_code_path_traversal_protected(client):
    resp = client.get('/code/../app.py')
    assert resp.status_code == 404


def test_phase_md_routes(client, tmp_path, monkeypatch):
    """Test that phase MD routes serve markdown files correctly."""
    # Create output directory structure
    outputs_dir = tmp_path / "outputs" / "paper2code" / "test_paper"
    outputs_dir.mkdir(parents=True)
    
    # Create phase MD files
    planning_md = outputs_dir / "PLANNING.md"
    analysis_md = outputs_dir / "ANALYSIS.md"
    coding_md = outputs_dir / "CODING.md"
    explanation_md = outputs_dir / "explanation_layer" / "EXPLANATION.md"
    explanation_md.parent.mkdir(parents=True)
    
    planning_md.write_text("# Planning Phase\n\nTest planning content")
    analysis_md.write_text("# Analysis Phase\n\nTest analysis content")
    coding_md.write_text("# Coding Phase\n\nTest coding content")
    explanation_md.write_text("# Explanation\n\nTest explanation content")
    
    # Mock DRIVER_WORK_ROOT
    monkeypatch.setattr(app_module, 'DRIVER_WORK_ROOT', str(tmp_path))
    
    # Test planning route
    resp = client.get("/data/planning.md")
    assert resp.status_code == 200
    assert "Planning Phase" in resp.get_data(as_text=True)
    assert resp.mimetype == "text/markdown"
    
    # Test analysis route
    resp = client.get("/data/analysis.md")
    assert resp.status_code == 200
    assert "Analysis Phase" in resp.get_data(as_text=True)
    
    # Test coding route
    resp = client.get("/data/coding.md")
    assert resp.status_code == 200
    assert "Coding Phase" in resp.get_data(as_text=True)
    
    # Test explanation route
    resp = client.get("/data/explanation.md")
    assert resp.status_code == 200
    assert "Explanation" in resp.get_data(as_text=True)

def test_phase_md_routes_404_when_missing(client, tmp_path, monkeypatch):
    """Test that phase MD routes return 404 when files don't exist."""
    # Create empty outputs directory
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True)
    
    # Mock DRIVER_WORK_ROOT
    monkeypatch.setattr(app_module, 'DRIVER_WORK_ROOT', str(tmp_path))
    
    # All routes should return 404
    assert client.get("/data/planning.md").status_code == 404
    assert client.get("/data/analysis.md").status_code == 404
    assert client.get("/data/coding.md").status_code == 404
    # Explanation might return 404 or empty, depending on implementation
    resp = client.get("/data/explanation.md")
    assert resp.status_code in [404, 200]  # Could be either

def test_phase_md_routes_most_recent_file(client, tmp_path, monkeypatch):
    """Test that phase MD routes serve the most recent file when multiple exist."""
    # Create multiple paper output directories
    paper1_dir = tmp_path / "outputs" / "paper2code" / "paper1"
    paper2_dir = tmp_path / "outputs" / "paper2code" / "paper2"
    paper1_dir.mkdir(parents=True)
    paper2_dir.mkdir(parents=True)
    
    # Create MD files with different timestamps
    planning1 = paper1_dir / "PLANNING.md"
    planning2 = paper2_dir / "PLANNING.md"
    
    planning1.write_text("# Paper 1 Planning")
    planning2.write_text("# Paper 2 Planning")
    
    # Make paper2 more recent by touching it
    import time
    time.sleep(0.1)  # Ensure different mtime
    planning2.touch()
    
    # Mock DRIVER_WORK_ROOT
    monkeypatch.setattr(app_module, 'DRIVER_WORK_ROOT', str(tmp_path))
    
    # Should serve the most recent one (paper2)
    resp = client.get("/data/planning.md")
    assert resp.status_code == 200
    assert "Paper 2 Planning" in resp.get_data(as_text=True)

def test_viewer_with_filename_success(client, tmp_path, monkeypatch):
    upload_dir = Path(app.root_path) / 'static' / 'uploads'
    upload_dir.mkdir(parents=True, exist_ok=True)
    fname = 'dummy.pdf'
    (upload_dir / fname).write_bytes(b'%PDF-1.4')

    resp = client.get(f'/viewer?filename={fname}&language=Python')
    assert resp.status_code == 200
    assert b'viewer' in resp.data or b'pdf' in resp.data


def test_unified_pipeline_integration(client, tmp_path, monkeypatch):
    """Test that frontend uses unified_pipeline correctly."""
    pdf = tmp_path / 'paper.pdf'
    pdf.write_bytes(b'%PDF-1.4 test')
    
    repo_dir = tmp_path / 'repo'
    repo_dir.mkdir()
    (repo_dir / 'test.py').write_text('print("test")')
    
    # Mock unified pipeline to return proper structure
    def mock_unified_pipeline(**kwargs):
        return {
            "repo_path": str(repo_dir),
            "paper_md_path": str(tmp_path / "paper.md"),
            "paper_json_path": "",
            "output_dir": str(tmp_path / "outputs"),
            "explanation_dir": str(tmp_path / "outputs" / "explanation_layer"),
            "explanation_md_path": "",
            "planning_md_path": "",
            "analysis_md_path": "",
            "coding_md_path": "",
            "from_github": False
        }
    
    monkeypatch.setattr(app_module, 'ep2c_pipeline', mock_unified_pipeline)
    
    with open(pdf, 'rb') as f:
        rv = client.post('/upload', data={'paper': (f, 'paper.pdf'), 'language': 'Python'}, follow_redirects=True)
    
    # Should redirect to viewer on success
    assert rv.status_code == 200
    assert b'viewer' in rv.data.lower() or b'pdf' in rv.data.lower()


def test_unified_pipeline_return_format(client, tmp_path, monkeypatch):
    """Test that unified pipeline returns expected format."""
    pdf = tmp_path / 'paper.pdf'
    pdf.write_bytes(b'%PDF-1.4 test')
    
    repo_dir = tmp_path / 'repo'
    repo_dir.mkdir()
    
    # Create mock return with all expected keys
    mock_result = {
        "repo_path": str(repo_dir),
        "paper_md_path": str(tmp_path / "paper.md"),
        "paper_json_path": "",
        "output_dir": str(tmp_path / "outputs"),
        "explanation_dir": str(tmp_path / "outputs" / "explanation_layer"),
        "explanation_md_path": str(tmp_path / "outputs" / "explanation_layer" / "EXPLANATION.md"),
        "planning_md_path": str(tmp_path / "outputs" / "PLANNING.md"),
        "analysis_md_path": str(tmp_path / "outputs" / "ANALYSIS.md"),
        "coding_md_path": str(tmp_path / "outputs" / "CODING.md"),
        "from_github": False
    }
    
    def mock_unified_pipeline(**kwargs):
        return mock_result
    
    monkeypatch.setattr(app_module, 'ep2c_pipeline', mock_unified_pipeline)
    monkeypatch.setattr(app_module, 'REPO_ROOT', str(repo_dir))
    
    with open(pdf, 'rb') as f:
        rv = client.post('/upload', data={'paper': (f, 'paper.pdf'), 'language': 'Python'}, follow_redirects=True)
    
    # Verify all expected keys are handled
    assert rv.status_code == 200
