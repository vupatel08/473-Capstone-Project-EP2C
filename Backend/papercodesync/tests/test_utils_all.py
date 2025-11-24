from pathlib import Path
import sys
import types
import tempfile
import json


HERE = Path(__file__).resolve().parent
SRC = str(HERE.parent / "src")
sys.path.insert(0, SRC)


cfg = {
    'utils': {
        'python': {'max_leading_lines': 5, 'stop_on_blank_line': True},
        'c_like': {'max_leading_lines': 4},
        'slugify_maxlen': 80,
        'id_truncate': 12,
        'token_pattern': r"[A-Za-z0-9_-]+",
        'text': {'min_paragraph_len': 3, 'chunk_max_chars': 20, 'chunk_hard_max_chars': 30, 'paragraph_join': '\n\n'},
        'latex': {'strip_space_after_commands': True, 'collapse_mathrm_inner_spaces': True, 'collapse_multi_spaces': True},
        'markdown': {'eq_fence': '$$', 'reference_headings': ['References'], 'fold_image_alt_into_prose': True},
    }
}

pc = types.ModuleType('utils.parse_config')
def _load_config(fp):
    return cfg
pc.load_config = _load_config
sys.modules['utils.parse_config'] = pc


from utils import comments, common, languages, latex, markdown


def test_comments_leading_hash_and_docblock():
    src = b"""
# first comment
# second comment

def foo():
    pass
"""
    node = types.SimpleNamespace(start_point=(4, 0), start_byte=0, end_byte=0)
    res = comments.leading_hash_comments_python(src, node)
    assert 'first comment' in res and 'second comment' in res

    src2 = b"""/**\n * hello\n */\nint x;\n"""
    node2 = types.SimpleNamespace(start_point=(4,0), start_byte=src2.find(b"int x"), end_byte=0)
    out = comments.leading_docblock_or_slashes(src2, node2)
    assert 'hello' in out

    src3 = b"""// one\n// two\nint main(){}\n"""
    node3 = types.SimpleNamespace(start_point=(3,0), start_byte=len(b"// one\n// two\n"), end_byte=0)
    out2 = comments.leading_docblock_or_slashes(src3, node3)
    assert 'one' in out2 and 'two' in out2


def test_common_text_and_chunking_and_hashes(tmp_path):
    assert common.slugify('Hello, World!') == 'hello-world'
    assert len(common.sha1_prefix('abc')) == common.ID_TRUNCATE
    assert common.keep_text('   ok  ') is False
    assert common.keep_text('  ') is False

    text = 'a' * 50
    chunks = common.split_into_chunks(text, max_chars=10, hard_max=12, joiner='\n')
    assert all(isinstance(c, str) for c in chunks)

    assert common.min_max_norm([5.0, 5.0]) == [0.0, 0.0]
    toks = common.tokenize('Hello WORLD_123!')
    assert 'hello' in toks and 'world_123' in toks

    rows = [{'a':1}, {'b':2}]
    p = tmp_path / 'out.jsonl'
    common.save_jsonl(rows, str(p))
    content = p.read_text(encoding='utf-8')
    assert '{"a": 1}' in content and '{"b": 2}' in content
    p2 = tmp_path / 'one.json'
    p2.write_text(json.dumps({'x': 1}))
    assert common.load_json(str(p2)) == {'x':1}

    assert common.read_bytes_safe(str(tmp_path / 'nope.bin')) == b''

    data = b'abcdefg'
    node = types.SimpleNamespace(start_byte=2, end_byte=5)
    assert common.slice_text(data, node) == b'cde'


def test_languages_and_latex_and_markdown():
    assert languages.EXT_TO_LANG['.py'] == 'python'

    s = '\\cmd  arg  \\mathrm{A B }  '
    out = latex.normalize_latex(s)
    assert '\\cmd' in out and 'mathrm' in out

    assert markdown.is_eq_fence('$$') is True
    assert markdown.is_reference_heading('References') is True


def test_common_additional_behaviors(tmp_path):
    assert isinstance(common.sha1_hex(b'abc'), str) and len(common.sha1_hex(b'abc')) == 40
    sid = common.sha1_id('a', 'b', prefix='p-')
    assert sid.startswith('p-')

    assert common.uniq_sorted(['b', 'a', 'b']) == ['a', 'b']

    assert common.normalize_text('  hi  ') == 'hi'
    assert common.safe_join_terms(['a', '', None, 'b']) == 'a b'

    p = tmp_path / 'sub' / 'file.txt'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('hello')
    assert common.posix_path(p) == p.as_posix()

    assert 'hello' in common.read_text_safe(str(p))

    assert common.read_bytes_safe(str(p)).startswith(b'hel')

    big = 'X' * 65
    parts = common.split_into_chunks(big, max_chars=20, hard_max=16, joiner='\n')
    assert all(len(part) <= 16 for part in parts)


def test_comments_stop_on_blank_and_c_max_lines():
    src = b"""
# keep me

# not-kept
int x;
"""
    node = types.SimpleNamespace(start_point=(4, 0), start_byte=src.find(b"int x"), end_byte=0)
    out = comments.leading_hash_comments_python(src, node)
    assert 'not-kept' in out and 'keep me' not in out

    pre = '\n'.join(['// a'] * 10) + '\nint y;\n'
    bpre = pre.encode('utf-8')
    node2 = types.SimpleNamespace(start_point=(11,0), start_byte=bpre.find(b'int y'), end_byte=0)
    out2 = comments.leading_docblock_or_slashes(bpre, node2)
    assert isinstance(out2, str)


def test_markdown_and_languages_more():
    m = markdown.HEADING_RE.match('# Title')
    assert m and m.group(1) == '#'
    im = markdown.IMG_RE.match('![alt](path.png)')
    assert im and im.group(1) == 'alt' and im.group(2) == 'path.png'

    for ext in ['.cpp', '.ts', '.jsx']:
        assert ext in languages.EXT_TO_LANG


def test_parse_config_loads_absolute(tmp_path):
    sample = {'foo': 'bar'}
    p = tmp_path / 'conf.yaml'
    import yaml
    p.write_text(yaml.safe_dump(sample))
    import importlib.util, importlib.machinery
    real_path = Path(__file__).resolve().parent.parent / 'src' / 'utils' / 'parse_config.py'
    loader = importlib.machinery.SourceFileLoader('real_parse_config', str(real_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    real_mod = importlib.util.module_from_spec(spec)
    loader.exec_module(real_mod)
    loaded = real_mod.load_config(str(p))
    assert loaded['foo'] == 'bar'


def test_comments_and_latex_edge_branches(tmp_path, monkeypatch):
    monkeypatch.setattr(comments, 'PY_STOP_ON_BLANK', False)
    src = b"""
# top

# middle
int z;
"""
    node = types.SimpleNamespace(start_point=(4,0), start_byte=src.find(b"int z"), end_byte=0)
    out = comments.leading_hash_comments_python(src, node)
    assert 'middle' in out

    bigpre = ('\n'.join(['// x'] * 10) + '\n/** doc */\nint v;').encode('utf-8')
    node2 = types.SimpleNamespace(start_point=(12,0), start_byte=bigpre.find(b'int v'), end_byte=0)
    out2 = comments.leading_docblock_or_slashes(bigpre, node2)
    assert 'doc' in out2

    assert latex.normalize_latex('') == ''


def test_common_uncovered_branches(tmp_path):
    assert common.slugify('') == ''

    s = 'para1\n\npara2\n\npara3'
    parts = common.split_into_chunks(s)
    assert isinstance(parts, list) and len(parts) >= 1

    parts2 = common.split_into_chunks('a||bbbb||cc', max_chars=3, joiner='||', hard_max=10)
    assert isinstance(parts2, list)

    assert common.tokenize('') == []

    assert common.min_max_norm([]) == []
    norm = common.min_max_norm([1.0, 3.0])
    assert norm[0] == 0.0 and norm[1] == 1.0


def test_parse_config_relative_load(tmp_path):
    root = Path(__file__).resolve().parent.parent
    p = root / 'src' / 'temp_rel.yaml'
    import yaml
    p.write_text(yaml.safe_dump({'k': 'v'}))
    try:
        import importlib.machinery, importlib.util
        real_path = Path(__file__).resolve().parent.parent / 'src' / 'utils' / 'parse_config.py'
        loader = importlib.machinery.SourceFileLoader('real_parse_config', str(real_path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        real_mod = importlib.util.module_from_spec(spec)
        loader.exec_module(real_mod)
        loaded = real_mod.load_config('temp_rel.yaml')
        assert loaded['k'] == 'v'
    finally:
        p.unlink()


def test_comments_additional_missing_branches():
    src = b"""
code line
# a comment
def f(): pass
"""
    node = types.SimpleNamespace(start_point=(4,0), start_byte=src.find(b"def f"), end_byte=0)
    out = comments.leading_hash_comments_python(src, node)
    assert out == ''

    src2 = b"random\nstuff\nint x\n"
    node2 = types.SimpleNamespace(start_point=(4,0), start_byte=len(src2), end_byte=0)
    out2 = comments.leading_docblock_or_slashes(src2, node2)
    assert out2 == ''
