from pathlib import Path
import sys
import types
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

if 'driver' not in sys.modules:
    _m = types.ModuleType('driver')
    def _fake_pipeline(*a, **k):
        raise RuntimeError('pipeline should not be invoked in template tests')
    _m.pcs_pipeline = _fake_pipeline
    _m.ep2c_pipeline = _fake_pipeline
    sys.modules['driver'] = _m

if 'pipeline' not in sys.modules:
    _m2 = types.ModuleType('pipeline')
    def _fake_run(*a, **k):
        return None
    _m2.run = _fake_run
    sys.modules['pipeline'] = _m2

import importlib
app_module = importlib.import_module('app')
app = app_module.app

def test_index_template_contains_form_and_js(client=None):
    try:
        from tests.test_app import client as client_fixture
    except Exception:
        client_fixture = None

    if client_fixture:
        with client_fixture() as c:
            resp = c.get('/')
    else:
        with app.test_client() as c:
            resp = c.get('/')

    assert resp.status_code == 200
    html = resp.data.decode('utf-8')

    assert '<form id="uploadForm"' in html
    assert 'input id="paper"' in html or 'name="paper"' in html
    assert 'select id="language"' in html

    assert 'label.addEventListener' in html or 'input.addEventListener' in html
    assert 'form.addEventListener' in html

    for lang in app_module.LANGUAGES:
        assert lang in html
