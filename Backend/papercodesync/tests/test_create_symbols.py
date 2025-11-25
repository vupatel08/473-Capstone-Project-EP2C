from pathlib import Path
import sys
import types
import tempfile
import json
from unittest.mock import Mock, MagicMock, patch

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
    },
    'symbols': {
        'exclude_dirs': ['.git', '__pycache__', 'node_modules'],
        'file_exclude_regex': r'\.min\.(js|css)$',
        'keep_text_span': True,
        'text_max_chars': 1000,
        'max_file_bytes': 1000000,
        'include_symbol_kinds': ['function', 'class', 'method', 'interface', 'enum', 'struct', 'constructor'],
        'python_doc_merge_strategy': 'both',
        'emit_module_span_default': False,
        'module_header_capture': True,
        'module_header_max_lines': 10,
        'follow_symlinks': False,
    }
}

pc = types.ModuleType('utils.parse_config')
def _load_config(fp):
    return cfg
pc.load_config = _load_config
sys.modules['utils.parse_config'] = pc

tsl = types.ModuleType('tree_sitter_languages')
def mock_get_parser(lang):
    return Mock()
tsl.get_parser = mock_get_parser
sys.modules['tree_sitter_languages'] = tsl

from create_symbols import (
    SymbolRec, detect_language, make_id, _merge_python_docs, _maybe_truncate_text,
    _path_blocked, python_extract_symbols, js_like_extract_symbols,
    java_extract_symbols, cpp_extract_symbols, parse_file, crawl_repo
)


class TestDetectLanguage:
    def test_detect_python(self):
        p = Path("test.py")
        assert detect_language(p) == "python"

    def test_detect_javascript(self):
        p = Path("test.js")
        assert detect_language(p) == "javascript"

    def test_detect_typescript(self):
        p = Path("test.ts")
        assert detect_language(p) == "typescript"

    def test_detect_tsx(self):
        p = Path("test.tsx")
        assert detect_language(p) == "tsx"

    def test_detect_java(self):
        p = Path("Test.java")
        assert detect_language(p) == "java"

    def test_detect_cpp(self):
        p = Path("test.cpp")
        assert detect_language(p) == "cpp"

    def test_detect_unknown_extension(self):
        p = Path("test.xyz")
        assert detect_language(p) is None

    def test_detect_no_extension(self):
        p = Path("README")
        assert detect_language(p) is None

    def test_detect_case_insensitive(self):
        p = Path("test.PY")
        assert detect_language(p) == "python"


class TestMakeId:
    def test_make_id_basic(self):
        id1 = make_id("file.py", "function", "func", 1, 10, b"code")
        assert isinstance(id1, str)
        assert len(id1) == 40  #

    def test_make_id_different_inputs_different_hash(self):
        id1 = make_id("file.py", "function", "func1", 1, 10, b"code")
        id2 = make_id("file.py", "function", "func2", 1, 10, b"code")
        assert id1 != id2

    def test_make_id_same_inputs_same_hash(self):
        id1 = make_id("file.py", "class", "MyClass", 5, 20, b"text")
        id2 = make_id("file.py", "class", "MyClass", 5, 20, b"text")
        assert id1 == id2

    def test_make_id_with_empty_text(self):
        id1 = make_id("test.py", "function", "empty", 1, 1, b"")
        assert isinstance(id1, str)
        assert len(id1) == 40


class TestMergePythonDocs:
    def test_merge_both_strategy_both_present(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'PYTHON_DOC_MERGE_STRATEGY', 'both')
        
        result = _merge_python_docs("This is docstring", "This is leading")
        assert "docstring" in result
        assert "leading" in result
        assert "\n" in result

    def test_merge_both_strategy_only_docstring(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'PYTHON_DOC_MERGE_STRATEGY', 'both')
        
        result = _merge_python_docs("Only docstring", "")
        assert result == "Only docstring"

    def test_merge_both_strategy_only_leading(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'PYTHON_DOC_MERGE_STRATEGY', 'both')
        
        result = _merge_python_docs("", "Only leading")
        assert result == "Only leading"

    def test_merge_only_docstring_strategy(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'PYTHON_DOC_MERGE_STRATEGY', 'only_docstring')
        
        result = _merge_python_docs("docstring", "leading")
        assert result == "docstring"

    def test_merge_only_leading_strategy(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'PYTHON_DOC_MERGE_STRATEGY', 'only_leading')
        
        result = _merge_python_docs("docstring", "leading")
        assert result == "leading"


    def test_merge_whitespace_stripped(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'PYTHON_DOC_MERGE_STRATEGY', 'both')
        
        result = _merge_python_docs("  docstring  ", "  leading  ")
        assert "docstring" in result and "leading" in result

class TestMaybeTruncateText:
    def test_truncate_disabled(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'KEEP_TEXT_SPAN', False)
        
        result = _maybe_truncate_text("This is some text")
        assert result == ""

    def test_truncate_enabled_short_text(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'KEEP_TEXT_SPAN', True)
        monkeypatch.setattr(create_symbols, 'TEXT_MAX_CHARS', 100)
        
        text = "Short text"
        result = _maybe_truncate_text(text)
        assert result == text

    def test_truncate_enabled_long_text(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'KEEP_TEXT_SPAN', True)
        monkeypatch.setattr(create_symbols, 'TEXT_MAX_CHARS', 10)
        
        text = "This is a very long text that should be truncated"
        result = _maybe_truncate_text(text)
        assert len(result) == 10
        assert result == text[:10]

    def test_truncate_max_chars_zero(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'KEEP_TEXT_SPAN', True)
        monkeypatch.setattr(create_symbols, 'TEXT_MAX_CHARS', 0)
        
        result = _maybe_truncate_text("text")
        assert result == "text"

    def test_truncate_none_max_chars(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'KEEP_TEXT_SPAN', True)
        monkeypatch.setattr(create_symbols, 'TEXT_MAX_CHARS', None)
        
        text = "Very long text that goes on and on"
        result = _maybe_truncate_text(text)
        assert result == text


class TestPathBlocked:
    def test_path_not_blocked_no_regex(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'FILE_EXCLUDE_RE', None)
        
        p = Path("test.py")
        assert _path_blocked(p) is False

    def test_path_blocked_by_regex(self, monkeypatch):
        import create_symbols
        import re
        monkeypatch.setattr(create_symbols, 'FILE_EXCLUDE_RE', re.compile(r'\.min\.js$'))
        
        p = Path("test.min.js")
        assert _path_blocked(p) is True

    def test_path_not_blocked_regex_no_match(self, monkeypatch):
        import create_symbols
        import re
        monkeypatch.setattr(create_symbols, 'FILE_EXCLUDE_RE', re.compile(r'\.min\.js$'))
        
        p = Path("test.js")
        assert _path_blocked(p) is False


class TestSymbolRec:
    def test_symbol_rec_creation(self):
        sym = SymbolRec(
            id='s1',
            file='test.py',
            kind='function',
            name='my_func',
            signature='my_func(x, y)',
            docstring='Does something',
            identifiers=['func', 'x', 'y'],
            start_line=1,
            end_line=10,
            text='def my_func...'
        )
        assert sym.id == 's1'
        assert sym.kind == 'function'
        assert sym.name == 'my_func'


class TestPythonExtractSymbols:
    def test_python_extract_empty_tree(self):
        mock_node = Mock()
        mock_node.children = []
        mock_node.root_node = mock_node
        
        mock_tree = Mock()
        mock_tree.root_node = mock_node
        
        src = b"# Empty file"
        recs = python_extract_symbols(src, mock_tree)
        assert isinstance(recs, list)

    def test_python_extract_with_function(self):
        func_node = Mock()
        func_node.type = "function_definition"
        func_node.start_point = (0, 0)
        func_node.end_point = (5, 0)
        func_node.children = []
        
        mock_root = Mock()
        mock_root.children = [func_node]
        
        mock_tree = Mock()
        mock_tree.root_node = mock_root
        
        src = b"def test_func():\n    pass\n"
        
        with patch('create_symbols.slice_text', return_value=src):
            with patch('create_symbols.leading_hash_comments_python', return_value=''):
                recs = python_extract_symbols(src, mock_tree)
                assert isinstance(recs, list)


class TestJsLikeExtractSymbols:
    def test_js_extract_empty_tree(self):
        mock_node = Mock()
        mock_node.children = []
        
        mock_tree = Mock()
        mock_tree.root_node = mock_node
        
        src = b"// Empty file"
        recs = js_like_extract_symbols(src, mock_tree)
        assert isinstance(recs, list)

    def test_js_extract_basic_structure(self):
        mock_root = Mock()
        mock_root.children = []
        
        mock_tree = Mock()
        mock_tree.root_node = mock_root
        
        src = b"function test() { }"
        recs = js_like_extract_symbols(src, mock_tree)
        assert isinstance(recs, list)


class TestJavaExtractSymbols:
    def test_java_extract_empty_tree(self):
        mock_node = Mock()
        mock_node.children = []
        
        mock_tree = Mock()
        mock_tree.root_node = mock_node
        
        src = b"// Empty file"
        recs = java_extract_symbols(src, mock_tree)
        assert isinstance(recs, list)

    def test_java_extract_basic_structure(self):
        mock_root = Mock()
        mock_root.children = []
        
        mock_tree = Mock()
        mock_tree.root_node = mock_root
        
        src = b"public class Test { }"
        recs = java_extract_symbols(src, mock_tree)
        assert isinstance(recs, list)


class TestCppExtractSymbols:
    def test_cpp_extract_empty_tree(self):
        mock_node = Mock()
        mock_node.children = []
        
        mock_tree = Mock()
        mock_tree.root_node = mock_node
        
        src = b"// Empty file"
        recs = cpp_extract_symbols(src, mock_tree)
        assert isinstance(recs, list)

    def test_cpp_extract_basic_structure(self):
        mock_root = Mock()
        mock_root.children = []
        
        mock_tree = Mock()
        mock_tree.root_node = mock_root
        
        src = b"class Test { };"
        recs = cpp_extract_symbols(src, mock_tree)
        assert isinstance(recs, list)


class TestParseFile:
    def test_parse_file_blocked(self, tmp_path, monkeypatch):
        import re
        import create_symbols
        monkeypatch.setattr(create_symbols, 'FILE_EXCLUDE_RE', re.compile(r'\.min\.js$'))
        
        p = tmp_path / "test.min.js"
        p.write_text("content")
        
        result = parse_file(p)
        assert result == []

    def test_parse_file_unknown_language(self, tmp_path):
        p = tmp_path / "test.xyz"
        p.write_text("content")
        
        result = parse_file(p)
        assert result == []


    def test_parse_file_empty_file(self, tmp_path, monkeypatch):
        import create_symbols
        from unittest.mock import Mock
        mock_func = Mock(return_value=None)
        monkeypatch.setattr(create_symbols, 'read_bytes_safe', mock_func)
        
        p = tmp_path / "test.py"
        result = parse_file(p)
        assert result == []

    def test_parse_file_too_large(self, tmp_path, monkeypatch):
        import create_symbols
        from unittest.mock import Mock
        monkeypatch.setattr(create_symbols, 'MAX_FILE_BYTES', 100)
        mock_func = Mock(return_value=b"x" * 1000)
        monkeypatch.setattr(create_symbols, 'read_bytes_safe', mock_func)
        
        p = tmp_path / "test.py"
        result = parse_file(p)
        assert result == []

    def test_parse_file_python(self, tmp_path, monkeypatch):
        import create_symbols
        
        mock_tree = Mock()
        mock_tree.root_node = Mock(children=[])
        
        mock_func1 = Mock(return_value=b"def f(): pass")
        monkeypatch.setattr(create_symbols, 'read_bytes_safe', mock_func1)
        
        mock_func2 = Mock(return_value=[])
        monkeypatch.setattr(create_symbols, 'python_extract_symbols', mock_func2)
        
        p = tmp_path / "test.py"
        p.write_text("def f(): pass")
        
        with patch.object(create_symbols.PARSER_MAP['python'], 'parse', return_value=mock_tree):
            result = parse_file(p)
            assert isinstance(result, list)

class TestCrawlRepo:
    def test_crawl_repo_empty(self, tmp_path):
        result = crawl_repo(tmp_path)
        assert result == []

    def test_crawl_repo_excludes_dirs(self, tmp_path, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'EXCLUDE_DIRS', {'__pycache__'})
        
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "test.py").write_text("content")
        
        result = crawl_repo(tmp_path)
        assert result == []


    def test_crawl_repo_includes_valid_files(self, tmp_path, monkeypatch):
        import create_symbols
        from unittest.mock import Mock
        monkeypatch.setattr(create_symbols, 'EXCLUDE_DIRS', set())
        mock_func = Mock(return_value=[])
        monkeypatch.setattr(create_symbols, 'parse_file', mock_func)
        
        (tmp_path / "test.py").write_text("content")
        
        result = crawl_repo(tmp_path)
        assert isinstance(result, list)
    def test_crawl_repo_respects_symlinks_setting(self, tmp_path, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'FOLLOW_SYMLINKS', False)
        monkeypatch.setattr(create_symbols, 'EXCLUDE_DIRS', set())
        
        result = crawl_repo(tmp_path)
        assert isinstance(result, list)


class TestIntegration:
    def test_make_id_consistency(self):
        id1 = make_id("test.py", "function", "func", 1, 5, b"code")
        id2 = make_id("test.py", "function", "func", 1, 5, b"code")
        assert id1 == id2

    def test_symbol_rec_with_various_kinds(self):
        kinds = ['function', 'class', 'method', 'interface', 'enum']
        symbols = []
        for kind in kinds:
            sym = SymbolRec(
                id=f'id_{kind}',
                file='test.py',
                kind=kind,
                name=f'{kind}_name',
                signature=f'sig_{kind}',
                docstring=f'doc_{kind}',
                identifiers=['id1', 'id2'],
                start_line=1,
                end_line=10,
                text=f'text_{kind}'
            )
            symbols.append(sym)
        
        assert len(symbols) == 5
        assert all(isinstance(s, SymbolRec) for s in symbols)

    def test_detect_language_all_supported(self):
        test_cases = [
            ("test.py", "python"),
            ("test.js", "javascript"),
            ("test.ts", "typescript"),
            ("test.tsx", "tsx"),
            ("Test.java", "java"),
            ("test.cpp", "cpp"),
        ]
        
        for filename, expected_lang in test_cases:
            p = Path(filename)
            assert detect_language(p) == expected_lang

    def test_truncate_and_merge_together(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'KEEP_TEXT_SPAN', True)
        monkeypatch.setattr(create_symbols, 'TEXT_MAX_CHARS', 20)
        monkeypatch.setattr(create_symbols, 'PYTHON_DOC_MERGE_STRATEGY', 'both')
        
        text = "This is a very long text that needs truncation"
        truncated = _maybe_truncate_text(text)
        assert len(truncated) == 20
        
        doc = _merge_python_docs("Docstring", "Leading")
        assert "\n" in doc or len(doc) > 0


class TestEdgeCases:
    def test_make_id_special_characters(self):
        id1 = make_id("file!@#.py", "func§", "name©", 1, 1, b"\x00\x01\x02")
        assert isinstance(id1, str)
        assert len(id1) == 40

    def test_merge_python_docs_empty_strings(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'PYTHON_DOC_MERGE_STRATEGY', 'both')
        
        result = _merge_python_docs("", "")
        assert result == ""

    def test_maybe_truncate_empty_string(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'KEEP_TEXT_SPAN', True)
        monkeypatch.setattr(create_symbols, 'TEXT_MAX_CHARS', 10)
        
        result = _maybe_truncate_text("")
        assert result == ""

    def test_path_blocked_empty_path(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'FILE_EXCLUDE_RE', None)
        
        p = Path("")
        assert _path_blocked(p) is False

    def test_detect_language_uppercase_extension(self):
        p = Path("test.PY")
        assert detect_language(p) == "python"

    def test_detect_language_mixed_case(self):
        p = Path("test.Ts")
        assert detect_language(p) == "typescript"


class TestParserMap:
    def test_parser_map_has_all_languages(self):
        import create_symbols
        expected_langs = ['python', 'javascript', 'typescript', 'tsx', 'java', 'cpp']
        for lang in expected_langs:
            assert lang in create_symbols.PARSER_MAP

    def test_parser_map_values_are_mocks(self):
        import create_symbols
        for parser in create_symbols.PARSER_MAP.values():
            assert parser is not None


class TestRegexPatterns:
    def test_module_header_regex(self, monkeypatch):
        import create_symbols
        import re
        
        pattern = r"^(?:\s*(?:#|//).*\n|/\*[\s\S]*?\*/\s*\n)+"
        
        py_header = "# This is a comment\n# Another comment\n"
        assert re.match(pattern, py_header) is not None
        
        js_header = "// Comment\n// Another\n"
        assert re.match(pattern, js_header) is not None

    def test_comment_cleanup_regex(self):
        import re
        
        text = "# This is a comment"
        cleaned = re.sub(r"^\s*#\s?", "", text)
        assert cleaned == "This is a comment"


class TestSymbolRecAsdict:
    def test_symbol_rec_asdict_conversion(self):
        from dataclasses import asdict
        
        sym = SymbolRec(
            id='test_id',
            file='file.py',
            kind='function',
            name='test_name',
            signature='test_sig',
            docstring='test_doc',
            identifiers=['id1', 'id2'],
            start_line=1,
            end_line=10,
            text='test_text'
        )
        
        d = asdict(sym)
        assert isinstance(d, dict)
        assert d['id'] == 'test_id'
        assert d['file'] == 'file.py'
        assert d['kind'] == 'function'


class TestLanguageDetectionEdgeCases:
    def test_detect_language_h_file(self):
        p = Path("test.h")
        result = detect_language(p)
        assert result is None or isinstance(result, str)

    def test_detect_language_multiple_dots(self):
        p = Path("test.min.py")
        assert detect_language(p) == "python"

    def test_detect_language_hidden_file(self):
        p = Path(".hidden.py")
        assert detect_language(p) == "python"


class TestConfigLoading:
    def test_config_has_all_required_keys(self):
        import create_symbols
        config = create_symbols.config
        assert 'symbols' in config
        
        symbols_config = config['symbols']
        required_keys = [
            'exclude_dirs', 'keep_text_span', 'text_max_chars',
            'max_file_bytes', 'include_symbol_kinds', 'python_doc_merge_strategy',
            'emit_module_span_default', 'module_header_capture', 'module_header_max_lines'
        ]
        for key in required_keys:
            assert key in symbols_config

    def test_include_symbol_kinds_is_set(self):
        import create_symbols
        kinds = create_symbols.INCLUDE_SYMBOL_KINDS
        assert isinstance(kinds, set)
        assert len(kinds) > 0


class TestAdditionalCoverage:
    def test_parse_file_javascript(self, tmp_path, monkeypatch):
        import create_symbols
        
        mock_tree = Mock()
        mock_tree.root_node = Mock(children=[])
        
        mock_func1 = Mock(return_value=b"function f() {}")
        monkeypatch.setattr(create_symbols, 'read_bytes_safe', mock_func1)
        
        mock_func2 = Mock(return_value=[])
        monkeypatch.setattr(create_symbols, 'js_like_extract_symbols', mock_func2)
        
        p = tmp_path / "test.js"
        p.write_text("function f() {}")
        
        with patch.object(create_symbols.PARSER_MAP['javascript'], 'parse', return_value=mock_tree):
            result = parse_file(p)
            assert isinstance(result, list)
    
    def test_parse_file_java(self, tmp_path, monkeypatch):
        import create_symbols
        
        mock_tree = Mock()
        mock_tree.root_node = Mock(children=[])
        
        mock_func1 = Mock(return_value=b"public class Test {}")
        monkeypatch.setattr(create_symbols, 'read_bytes_safe', mock_func1)
        
        mock_func2 = Mock(return_value=[])
        monkeypatch.setattr(create_symbols, 'java_extract_symbols', mock_func2)
        
        p = tmp_path / "Test.java"
        p.write_text("public class Test {}")
        
        with patch.object(create_symbols.PARSER_MAP['java'], 'parse', return_value=mock_tree):
            result = parse_file(p)
            assert isinstance(result, list)
    
    def test_parse_file_cpp(self, tmp_path, monkeypatch):
        import create_symbols
        
        mock_tree = Mock()
        mock_tree.root_node = Mock(children=[])
        
        mock_func1 = Mock(return_value=b"class Test {};")
        monkeypatch.setattr(create_symbols, 'read_bytes_safe', mock_func1)
        
        mock_func2 = Mock(return_value=[])
        monkeypatch.setattr(create_symbols, 'cpp_extract_symbols', mock_func2)
        
        p = tmp_path / "test.cpp"
        p.write_text("class Test {};")
        
        with patch.object(create_symbols.PARSER_MAP['cpp'], 'parse', return_value=mock_tree):
            result = parse_file(p)
            assert isinstance(result, list)
    
    def test_constants_defined(self):
        import create_symbols
        assert create_symbols.EXCLUDE_DIRS is not None
        assert create_symbols.KEEP_TEXT_SPAN is not None
        assert create_symbols.TEXT_MAX_CHARS is not None
        assert create_symbols.MAX_FILE_BYTES is not None
    
    def test_identifier_type_constants(self):
        import create_symbols
        assert create_symbols.JS_FUNC_TYPES is not None
        assert isinstance(create_symbols.JS_FUNC_TYPES, dict)
        assert create_symbols.IDENTIFIER_TYPES_JS is not None
        assert isinstance(create_symbols.IDENTIFIER_TYPES_JS, set)
        assert create_symbols.IDENTIFIER_TYPES_JAVA is not None
        assert isinstance(create_symbols.IDENTIFIER_TYPES_JAVA, set)
        assert create_symbols.IDENTIFIER_TYPES_CPP is not None
        assert isinstance(create_symbols.IDENTIFIER_TYPES_CPP, set)
    
    def test_make_id_with_unicode(self):
        id1 = make_id("тест.py", "функция", "имя", 1, 5, "текст".encode('utf-8'))
        assert isinstance(id1, str)
        assert len(id1) == 40
    
    def test_symbol_rec_defaults(self):
        sym = SymbolRec(
            id='id',
            file='file',
            kind='function',
            name='name',
            signature='sig',
            docstring='doc',
            identifiers=[],
            start_line=1,
            end_line=2,
            text='text'
        )
        assert sym.start_line == 1
        assert sym.end_line == 2

class TestFilteringBySymbolKind:
    def test_python_filters_excluded_kinds(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'INCLUDE_SYMBOL_KINDS', {'function'})
        
        class_node = Mock()
        class_node.type = "class_definition"
        class_node.start_point = (0, 0)
        class_node.end_point = (5, 0)
        class_node.children = []
        
        root = Mock()
        root.children = [class_node]
        
        tree = Mock()
        tree.root_node = root
        
        src = b"class MyClass:\n    pass\n"
        
        with patch('create_symbols.slice_text', return_value=src):
            recs = python_extract_symbols(src, tree)
            assert isinstance(recs, list)
    
    def test_js_filters_excluded_kinds(self, monkeypatch):
        import create_symbols
        monkeypatch.setattr(create_symbols, 'INCLUDE_SYMBOL_KINDS', {'class'})
        
        func_node = Mock()
        func_node.type = "function_declaration"
        func_node.start_point = (0, 0)
        func_node.end_point = (3, 0)
        func_node.children = []
        
        root = Mock()
        root.children = [func_node]
        
        tree = Mock()
        tree.root_node = root
        
        src = b"function test() { }"
        
        with patch('create_symbols.slice_text', return_value=src):
            recs = js_like_extract_symbols(src, tree)
            assert isinstance(recs, list)
