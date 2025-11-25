from pathlib import Path
import os
import sys
import types
import shutil
from unittest.mock import Mock
import json

HERE = Path(__file__).resolve().parent
SRC = str(HERE.parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import driver


def test_delete_files_in_directory(tmp_path, capsys):
    d = tmp_path / "data"
    d.mkdir()
    f1 = d / "a.txt"
    f2 = d / "b.log"
    f1.write_text("x")
    f2.write_text("y")
    (d / "subdir").mkdir()

    driver.delete_files_in_directory(str(d))

    captured = capsys.readouterr()
    assert not f1.exists()
    assert not f2.exists()
    assert (d / "subdir").exists()
    assert "Deleted file" in captured.out


def test_run_cmd_invokes_subprocess(monkeypatch, capsys):
    mock_run = Mock()
    monkeypatch.setattr(driver, 'subprocess', types.SimpleNamespace(run=mock_run))

    cmd = [sys.executable, "-c", "print('hi')"]
    cwd = Path("/tmp")
    driver.run_cmd(cmd, cwd=cwd)

    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert isinstance(args[0], list)
    assert kwargs.get('cwd') == str(cwd)
    assert kwargs.get('check') is True or 'check' in kwargs
    out = capsys.readouterr().out
    assert "$" in out


def test_ensure_exists_raises_and_ok(tmp_path, capsys):
    p = tmp_path / "nope"
    try:
        driver.ensure_exists(p, "Thing")
        assert False, "should have raised"
    except FileNotFoundError:
        pass

    f = tmp_path / "ok.txt"
    f.write_text("ok")
    driver.ensure_exists(f, "Thing")
    out = capsys.readouterr().out
    assert "[OK] Thing" in out


def test_pcs_pipeline_success(tmp_path, monkeypatch):
    md = tmp_path / "paper.md"
    md.write_text("# Title\ncontent")
    repo = tmp_path / "repo"
    repo.mkdir()

    monkeypatch.setattr(driver, 'DATA_DIR', tmp_path / 'data')
    driver.DATA_DIR.mkdir()
    monkeypatch.setattr(driver, 'CHUNKS_JSON', driver.DATA_DIR / 'chunks.json')
    monkeypatch.setattr(driver, 'SYMBOLS_JSON', driver.DATA_DIR / 'symbols.json')
    monkeypatch.setattr(driver, 'MATCHES_JSON', driver.DATA_DIR / 'matches.jsonl')

    create_chunks = tmp_path / 'create_chunks.py'
    create_chunks.write_text('')
    create_symbols = tmp_path / 'create_symbols.py'
    create_symbols.write_text('')
    create_map = tmp_path / 'create_map.py'
    create_map.write_text('')

    monkeypatch.setattr(driver, 'CREATE_CHUNKS', create_chunks)
    monkeypatch.setattr(driver, 'CREATE_SYMBOLS', create_symbols)
    monkeypatch.setattr(driver, 'CREATE_MAP', create_map)

    tmp_here = tmp_path / 'here'
    tmp_here.mkdir()
    monkeypatch.setattr(driver, 'HERE', tmp_here)

    def fake_run_cmd(cmd, cwd=None):
        if str(create_chunks) in cmd[1]:
            driver.CHUNKS_JSON.write_text(json.dumps([{"id": "c1", "text": "t"}]))
        if str(create_symbols) in cmd[1]:
            driver.SYMBOLS_JSON.write_text(json.dumps([{"id": "s1", "name": "n"}]))
        if str(create_map) in cmd[1]:
            (driver.HERE / 'matches.jsonl').write_text('{"a":1}\n')
    
    monkeypatch.setattr(driver, 'run_cmd', fake_run_cmd)

    res = driver.pcs_pipeline(md, repo)
    assert res is True
    assert driver.MATCHES_JSON.exists()


def test_pcs_pipeline_missing_md(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    try:
        driver.pcs_pipeline(tmp_path / 'no.md', repo)
        assert False
    except FileNotFoundError:
        pass


def test_delete_files_ignores_non_files(tmp_path):
    d = tmp_path / 'd'
    d.mkdir()
    (d / 'file.txt').write_text('x')
    (d / 'sub').mkdir()
    driver.delete_files_in_directory(str(d))
    assert not (d / 'file.txt').exists()
    assert (d / 'sub').exists()
