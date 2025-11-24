from pathlib import Path
import sys
import types
import importlib

HERE = Path(__file__).resolve().parent
PKG_ROOT = str(HERE.parent)
sys.path.insert(0, PKG_ROOT)

if 'gradio_client' not in sys.modules:
    fake_gc = types.ModuleType('gradio_client')
    class FakeClient:
        _next_return = None
        def __init__(self, *a, **k):
            pass
        def predict(self, *a, **k):
            return FakeClient._next_return
    fake_gc.Client = FakeClient
    sys.modules['gradio_client'] = fake_gc

import json
import find_repo


def test__normalize_various_shapes():
    r = {'paper': 'purl', 'name': 'pname', 'code': 'crepo'}
    assert find_repo._normalize(r) == ('purl', 'pname', 'crepo')

    s = json.dumps({'paper': 'p2', 'name': 'n2', 'code': 'c2'})
    assert find_repo._normalize(s) == ('p2', 'n2', 'c2')

    lst = ['paperA', 'nameA', 'codeA']
    assert find_repo._normalize(lst) == ('paperA', 'codeA', 'nameA')

    assert find_repo._normalize(12345) == ('', '', '')


def test_find_paper_uses_client_predict(monkeypatch):
    fake = sys.modules['gradio_client'].Client
    fake._next_return = {'paper': 'pX', 'name': 'nX', 'code': 'cX'}
    paper_url, paper_name, code_repo = find_repo.find_paper('ignored')
    assert paper_url == 'pX' and paper_name == 'nX' and code_repo == 'cX'

    fake._next_return = json.dumps({'paper': 'pY', 'name': 'nY', 'code': 'cY'})
    paper_url, paper_name, code_repo = find_repo.find_paper('ignored')
    assert paper_url == 'pY' and paper_name == 'nY' and code_repo == 'cY'

    fake._next_return = ['pL', 'nL', 'cL']
    paper_url, paper_name, code_repo = find_repo.find_paper('ignored')
    assert paper_url == 'pL' and paper_name == 'cL' and code_repo == 'nL'


def test_get_repo_link_behaviour(monkeypatch, capsys):
    monkeypatch.setattr(find_repo, 'find_link', lambda p: None)
    assert find_repo.get_repo_link('somepath') is None
    captured = capsys.readouterr()
    assert 'Could not find paper link.' in captured.out

    monkeypatch.setattr(find_repo, 'find_link', lambda p: 'paperlink')
    monkeypatch.setattr(find_repo, 'find_paper', lambda x: (None, None, None))
    assert find_repo.get_repo_link('somepath') is None
    captured = capsys.readouterr()
    assert 'Could not find paper information from MCP.' in captured.out

    monkeypatch.setattr(find_repo, 'find_paper', lambda x: ('purl', 'pname', ''))
    assert find_repo.get_repo_link('somepath') is None
    captured = capsys.readouterr()
    assert 'Could not find code repository' in captured.out

    monkeypatch.setattr(find_repo, 'find_paper', lambda x: ('purl', 'pname', 'http://repo'))
    res = find_repo.get_repo_link('somepath')
    assert res == 'http://repo'
    captured = capsys.readouterr()
    assert 'Found code repository for:' in captured.out
