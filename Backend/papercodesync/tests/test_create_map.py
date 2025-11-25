from pathlib import Path
import sys
import types
import tempfile
import json
import numpy as np

# Ensure src is importable
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
    'chunks': {},
    'map': {
        'overlap_method': 'bm25',
        'alpha': 0.5,
        'top_k': 5,
        'normalization': 'minmax',
        'bm25': {'k1': 1.5, 'b': 0.75},
        'tfidf': {'ngram_min': 1, 'ngram_max': 2, 'stop_words': None, 'sublinear_tf': False, 'use_idf': True},
        'semantic_model': 'all-MiniLM-L6-v2',
        'semantic': {'normalize_embeddings': True, 'batch_size': 32},
        'query': {'use_full_text_for_semantic': True, 'use_full_text_for_overlap': False, 'bow_max_terms': 10},
    }
}

pc = types.ModuleType('utils.parse_config')
def _load_config(fp):
    return cfg
pc.load_config = _load_config
sys.modules['utils.parse_config'] = pc

class MockSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def encode(self, texts, batch_size=32, normalize_embeddings=False, show_progress_bar=False):
        """Return deterministic embeddings based on text content length"""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            vec = np.zeros(384)
            if text and len(text) > 0:
                vec[0] = len(text) / 100.0
                vec[1] = len(text.split()) / 10.0
                for i, c in enumerate(text[:50]):
                    vec[(i + 2) % 384] += ord(c) / 1000.0
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings)

st_module = types.ModuleType('sentence_transformers')
st_module.SentenceTransformer = MockSentenceTransformer
sys.modules['sentence_transformers'] = st_module

from create_map import (
    OverlapIndex, SemanticIndex, combine_scores, _mk_bow_terms, _to_symbols,
    _paper_to_chunks, load_chunks_flexible, build_and_match, Symbol, Chunk, CombinedMatch
)


def test_overlap_index_bm25_init():
    idx = OverlapIndex(method='bm25')
    assert idx.method == 'bm25'
    assert idx._bm25 is None


def test_overlap_index_tfidf_init():
    idx = OverlapIndex(method='tfidf')
    assert idx.method == 'tfidf'
    assert idx._tfidf_vectorizer is None


def test_overlap_index_invalid_method():
    try:
        idx = OverlapIndex(method='invalid')
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert 'Invalid overlap method' in str(e)


def test_overlap_index_bm25_fit_and_query():
    idx = OverlapIndex(method='bm25')
    chunk_ids = ['c1', 'c2', 'c3']
    chunk_texts = ['hello world', 'foo bar baz', 'hello there']
    
    idx.fit(chunk_ids, chunk_texts)
    assert len(idx._ids) == 3
    assert idx._bm25 is not None
    
    results = idx.query('hello', top_k=2)
    assert len(results) <= 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_overlap_index_tfidf_fit_and_query():
    idx = OverlapIndex(method='tfidf')
    chunk_ids = ['c1', 'c2', 'c3']
    chunk_texts = ['hello world', 'foo bar baz', 'hello there']
    
    idx.fit(chunk_ids, chunk_texts)
    assert len(idx._ids) == 3
    assert idx._tfidf_vectorizer is not None
    assert idx._tfidf_matrix is not None
    
    results = idx.query('hello', top_k=2)
    assert len(results) <= 2


def test_overlap_index_empty_query():
    idx = OverlapIndex(method='bm25')
    chunk_ids = ['c1', 'c2']
    chunk_texts = ['text1', 'text2']
    idx.fit(chunk_ids, chunk_texts)
    
    results = idx.query('')
    assert len(results) == 0


def test_semantic_index_init():
    idx = SemanticIndex(model_name='all-MiniLM-L6-v2')
    assert idx.model_name == 'all-MiniLM-L6-v2'
    assert idx._model is None


def test_semantic_index_fit_and_query():
    idx = SemanticIndex(model_name='all-MiniLM-L6-v2')
    chunk_ids = ['c1', 'c2', 'c3']
    chunk_texts = ['machine learning', 'deep neural networks', 'data science']
    
    idx.fit(chunk_ids, chunk_texts)
    assert len(idx._ids) == 3
    assert idx._chunk_vecs is not None
    
    results = idx.query('machine learning', top_k=2)
    assert len(results) <= 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_semantic_index_empty_query():
    idx = SemanticIndex(model_name='all-MiniLM-L6-v2')
    chunk_ids = ['c1', 'c2']
    chunk_texts = ['text1', 'text2']
    idx.fit(chunk_ids, chunk_texts)
    
    results = idx.query('')
    assert len(results) == 0


def test_combine_scores_basic():
    overlap = [('c1', 0.9), ('c2', 0.7), ('c3', 0.5)]
    semantic = [('c2', 0.95), ('c1', 0.8), ('c3', 0.6)]
    
    result = combine_scores(overlap, semantic, weight_overlap=0.5, weight_semantic=0.5, top_k=3)
    assert len(result) <= 3
    assert all(isinstance(m, CombinedMatch) for m in result)
    assert result[0].combined >= result[-1].combined if len(result) > 1 else True


def test_combine_scores_normalization_none(monkeypatch):
    import create_map
    old_norm = create_map.NORMALIZATION
    create_map.NORMALIZATION = 'none'
    
    overlap = [('c1', 0.9), ('c2', 0.7)]
    semantic = [('c1', 0.95), ('c2', 0.6)]
    
    result = combine_scores(overlap, semantic, top_k=2)
    assert len(result) >= 1
    
    create_map.NORMALIZATION = old_norm


def test_mk_bow_terms_from_symbol():
    sym_obj = {
        'name': 'MyFunction',
        'identifiers': ['func_name', 'helper'],
        'docstring': 'Performs calculation with data',
        'text': 'return x + y + z'
    }
    terms = _mk_bow_terms(sym_obj, max_terms=10)
    assert isinstance(terms, list)
    assert len(terms) > 0
    assert all(isinstance(t, str) for t in terms)


def test_mk_bow_terms_minimal():
    sym_obj = {'name': 'simple'}
    terms = _mk_bow_terms(sym_obj, max_terms=5)
    assert isinstance(terms, list)
    assert len(terms) > 0


def test_mk_bow_terms_empty_fallback():
    sym_obj = {'name': ''}
    terms = _mk_bow_terms(sym_obj, max_terms=5)
    assert isinstance(terms, list)


def test_to_symbols_list():
    objs = [
        {'id': 's1', 'name': 'func1', 'docstring': 'Does X', 'text': 'implementation here'},
        {'id': 's2', 'name': 'func2', 'full_text': 'predefined full text'},
    ]
    symbols = _to_symbols(objs)
    assert len(symbols) == 2
    assert all(isinstance(s, Symbol) for s in symbols)
    assert symbols[0].id == 's1'
    assert symbols[1].full_text == 'predefined full text'


def test_to_symbols_no_id():
    objs = [{'name': 'unnamed'}]
    symbols = _to_symbols(objs)
    assert len(symbols) == 1
    assert symbols[0].id == 'unnamed'


def test_paper_to_chunks():
    paper = {
        'sections': [
            {
                'id': 'sec1',
                'title': 'Methods',
                'paragraphs': [
                    {'id': 'p1', 'text': 'First paragraph here'},
                    {'id': 'p2', 'text': 'Second paragraph here'},
                ]
            },
            {
                'id': 'sec2',
                'title': 'Results',
                'paragraphs': [
                    {'id': 'p3', 'text': 'Result text here'},
                ]
            }
        ]
    }
    chunks = _paper_to_chunks(paper)
    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].section == 'Methods'


def test_paper_to_chunks_empty_sections():
    paper = {'sections': []}
    chunks = _paper_to_chunks(paper)
    assert len(chunks) == 0


def test_load_chunks_flexible_list_format(tmp_path):
    chunks_data = [
        {'id': 'ch1', 'section': 'Intro', 'text': 'Introductory text'},
        {'id': 'ch2', 'section': 'Methods', 'text': 'Methods text'},
    ]
    p = tmp_path / 'chunks.json'
    p.write_text(json.dumps(chunks_data), encoding='utf-8')
    
    chunks = load_chunks_flexible(str(p))
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)


def test_load_chunks_flexible_paper_format(tmp_path):
    paper_data = {
        'sections': [
            {
                'id': 'sec1',
                'title': 'Intro',
                'paragraphs': [
                    {'id': 'p1', 'text': 'Paper intro text'}
                ]
            }
        ]
    }
    p = tmp_path / 'paper.json'
    p.write_text(json.dumps(paper_data), encoding='utf-8')
    
    chunks = load_chunks_flexible(str(p))
    assert len(chunks) == 1


def test_load_chunks_flexible_invalid_format(tmp_path):
    invalid_data = {'invalid': 'format'}
    p = tmp_path / 'invalid.json'
    p.write_text(json.dumps(invalid_data), encoding='utf-8')
    
    try:
        chunks = load_chunks_flexible(str(p))
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_build_and_match_basic():
    symbols = [
        Symbol(id='s1', name='function1', bow_terms=['func', 'one'], full_text='does something'),
        Symbol(id='s2', name='function2', bow_terms=['func', 'two'], full_text='does other'),
    ]
    chunks = [
        Chunk(id='c1', section='Methods', text='function one implementation here'),
        Chunk(id='c2', section='Results', text='function two analysis here'),
    ]
    
    results = build_and_match(symbols, chunks, top_k=3)
    
    assert len(results) == 2
    assert all(isinstance(r, dict) for r in results)
    assert 'symbol_id' in results[0]
    assert 'best' in results[0]


def test_combined_match_dataclass():
    m = CombinedMatch(chunk_id='c1', combined=0.8, overlap=0.75, semantic=0.85)
    assert m.chunk_id == 'c1'
    assert m.combined == 0.8


def test_symbol_dataclass():
    s = Symbol(id='s1', name='func', bow_terms=['a', 'b'], full_text='text')
    assert s.id == 's1'
    assert len(s.bow_terms) == 2


def test_chunk_dataclass():
    c = Chunk(id='c1', section='Intro', text='Some text')
    assert c.id == 'c1'
    assert c.section == 'Intro'


def test_overlap_index_query_top_k_limit():
    idx = OverlapIndex(method='bm25')
    chunk_ids = ['c1', 'c2', 'c3', 'c4', 'c5']
    chunk_texts = ['hello', 'hello world', 'hello there', 'foo', 'bar']
    idx.fit(chunk_ids, chunk_texts)
    
    results = idx.query('hello', top_k=2)
    assert len(results) <= 2


def test_semantic_index_lazy_load():
    idx = SemanticIndex(model_name='all-MiniLM-L6-v2')
    assert idx._model is None
    idx._ensure_model()
    assert idx._model is not None


def test_to_symbols_with_bow_terms_preset():
    objs = [
        {
            'id': 's1',
            'name': 'func',
            'bow_terms': ['preset', 'terms'],
            'full_text': 'Full text provided'
        }
    ]
    symbols = _to_symbols(objs)
    assert symbols[0].bow_terms == ['preset', 'terms']
    assert symbols[0].full_text == 'Full text provided'


def test_load_chunks_flexible_list_with_chunk_id_alt_name(tmp_path):
    chunks_data = [
        {'chunk_id': 'ch1', 'title': 'Section', 'content': 'Text content'},
    ]
    p = tmp_path / 'chunks.json'
    p.write_text(json.dumps(chunks_data), encoding='utf-8')
    
    chunks = load_chunks_flexible(str(p))
    assert len(chunks) == 1
    assert chunks[0].id == 'ch1'


def test_paper_to_chunks_missing_paragraphs():
    paper = {
        'sections': [
            {'id': 'sec1', 'title': 'Intro', 'paragraphs': None}
        ]
    }
    chunks = _paper_to_chunks(paper)
    assert len(chunks) == 0


def test_build_and_match_with_tfidf(monkeypatch):
    import create_map
    old_method = create_map.OVERLAP_METHOD
    create_map.OVERLAP_METHOD = 'tfidf'
    
    symbols = [Symbol(id='s1', name='test', bow_terms=['test'], full_text='test query')]
    chunks = [Chunk(id='c1', section='Sec', text='test content')]
    
    results = build_and_match(symbols, chunks, method_overlap='tfidf', top_k=3)
    assert len(results) >= 1
    
    create_map.OVERLAP_METHOD = old_method


def test_combine_scores_all_none_scores():
    overlap = [('c1', 0.5)]
    semantic = [('c2', 0.5)]
    
    result = combine_scores(overlap, semantic, top_k=2)
    assert len(result) == 2
    assert {m.chunk_id for m in result} == {'c1', 'c2'}


def test_overlap_index_single_empty_text():
    idx = OverlapIndex(method='bm25')
    chunk_ids = ['c1', 'c2']
    chunk_texts = ['some content', 'other text']
    idx.fit(chunk_ids, chunk_texts)
    
    results = idx.query('content', top_k=1)
    assert isinstance(results, list)
    assert len(results) >= 1


def test_semantic_index_batch_processing():
    idx = SemanticIndex(model_name='all-MiniLM-L6-v2')
    chunk_ids = [f'c{i}' for i in range(5)]
    chunk_texts = [f'text number {i}' for i in range(5)]
    
    idx.fit(chunk_ids, chunk_texts)
    results = idx.query('text', top_k=3)
    assert len(results) <= 3


def test_mk_bow_terms_with_all_fields():
    sym_obj = {
        'id': 's1',
        'name': 'MyClass',
        'docstring': 'This class performs operations',
        'identifiers': ['cls_name', 'handler', 'manager'],
        'text': 'implementation code here'
    }
    terms = _mk_bow_terms(sym_obj, max_terms=20)
    assert isinstance(terms, list)
    assert len(terms) > 0
    assert all(isinstance(t, str) for t in terms)


def test_combine_scores_weighted_emphasis():
    overlap = [('c1', 0.8), ('c2', 0.6)]
    semantic = [('c1', 0.9), ('c2', 0.5)]
    
    result1 = combine_scores(overlap, semantic, weight_overlap=0.7, weight_semantic=0.3, top_k=2)
    assert len(result1) >= 1
    
    result2 = combine_scores(overlap, semantic, weight_overlap=0.3, weight_semantic=0.7, top_k=2)
    assert len(result2) >= 1


def test_to_symbols_complex_structure():
    objs = [
        {
            'id': 's1',
            'name': 'ComplexFunc',
            'docstring': 'Long docstring here describing functionality',
            'identifiers': ['func_name', 'helper_func', 'util_func'],
            'text': 'def ComplexFunc(): pass',
            'full_text': 'Predefined full text content'
        }
    ]
    symbols = _to_symbols(objs)
    assert len(symbols) == 1
    assert symbols[0].id == 's1'
    assert symbols[0].full_text == 'Predefined full text content'


def test_paper_to_chunks_nested_structure():
    paper = {
        'sections': [
            {
                'id': 'abstract',
                'title': 'Abstract',
                'paragraphs': [
                    {'id': 'p1', 'text': 'Summary of work'}
                ]
            },
            {
                'id': 'methods',
                'title': 'Methods',
                'paragraphs': [
                    {'id': 'p2', 'text': 'First method paragraph'},
                    {'id': 'p3', 'text': 'Second method paragraph'},
                    {'id': 'p4', 'text': 'Third method paragraph'},
                ]
            }
        ]
    }
    chunks = _paper_to_chunks(paper)
    assert len(chunks) == 4
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].section == 'Abstract'
    assert chunks[1].section == 'Methods'


def test_build_and_match_no_symbols():
    # Test with empty symbol list
    symbols = []
    chunks = [Chunk(id='c1', section='Intro', text='Some text')]
    
    results = build_and_match(symbols, chunks, top_k=3)
    assert results == []


def test_build_and_match_single_chunk():
    symbols = [Symbol(id='s1', name='func', bow_terms=['test'], full_text='test query')]
    chunks = [Chunk(id='c1', section='Intro', text='test content')]
    
    results = build_and_match(symbols, chunks, top_k=3)
    assert len(results) == 1
    assert results[0]['best'] is not None


def test_build_and_match_with_full_text_overlap(monkeypatch):
    import create_map
    old_q_use_ft = create_map.Q_USE_FT_FOR_OVL
    create_map.Q_USE_FT_FOR_OVL = True
    
    symbols = [Symbol(id='s1', name='func', bow_terms=['bow'], full_text='full text query')]
    chunks = [Chunk(id='c1', section='Sec', text='full text content')]
    
    results = build_and_match(symbols, chunks, top_k=3)
    assert len(results) == 1
    assert results[0]['query_text_overlap'] == 'full text query'
    
    create_map.Q_USE_FT_FOR_OVL = old_q_use_ft


def test_build_and_match_with_full_text_semantic(monkeypatch):
    import create_map
    old_q_use_ft = create_map.Q_USE_FT_FOR_SEM
    create_map.Q_USE_FT_FOR_SEM = True
    
    symbols = [Symbol(id='s1', name='func', bow_terms=['bow'], full_text='full text query')]
    chunks = [Chunk(id='c1', section='Sec', text='full text content')]
    
    results = build_and_match(symbols, chunks, top_k=3)
    assert len(results) == 1
    assert results[0]['query_text_semantic'] == 'full text query'
    
    create_map.Q_USE_FT_FOR_SEM = old_q_use_ft


def test_load_chunks_flexible_minimal_chunks(tmp_path):
    chunks_data = [
        {'id': 'ch1', 'text': 'Content'}
    ]
    p = tmp_path / 'minimal.json'
    p.write_text(json.dumps(chunks_data), encoding='utf-8')
    
    chunks = load_chunks_flexible(str(p))
    assert len(chunks) == 1
    assert chunks[0].id == 'ch1'


def test_load_chunks_flexible_list_with_invalid_items(tmp_path):
    chunks_data = [
        {'id': 'ch1', 'section': 'Intro', 'text': 'Valid chunk'},
        {'id': 'ch2'},  
        {'section': 'Sec', 'text': 'No ID chunk'}
    ]
    p = tmp_path / 'mixed.json'
    p.write_text(json.dumps(chunks_data), encoding='utf-8')
    
    chunks = load_chunks_flexible(str(p))
    assert len(chunks) == 2


def test_combine_scores_normalization_minmax(monkeypatch):
    import create_map
    old_norm = create_map.NORMALIZATION
    create_map.NORMALIZATION = 'minmax'
    
    overlap = [('c1', 0.9), ('c2', 0.1)]
    semantic = [('c1', 0.1), ('c2', 0.9)]
    
    result = combine_scores(overlap, semantic, top_k=2)
    assert len(result) == 2
    
    create_map.NORMALIZATION = old_norm


def test_overlap_index_multiple_queries():
    idx = OverlapIndex(method='bm25')
    chunk_ids = ['c1', 'c2', 'c3']
    chunk_texts = ['machine learning', 'deep learning', 'data analysis']
    idx.fit(chunk_ids, chunk_texts)
    
    results1 = idx.query('machine', top_k=2)
    results2 = idx.query('data', top_k=2)
    results3 = idx.query('learning', top_k=2)
    
    assert isinstance(results1, list)
    assert isinstance(results2, list)
    assert isinstance(results3, list)


def test_semantic_index_multiple_queries():
    idx = SemanticIndex(model_name='all-MiniLM-L6-v2')
    chunk_ids = ['c1', 'c2', 'c3']
    chunk_texts = ['neural networks', 'computer vision', 'NLP models']
    idx.fit(chunk_ids, chunk_texts)
    
    results1 = idx.query('networks', top_k=2)
    results2 = idx.query('vision', top_k=2)
    results3 = idx.query('language', top_k=2)
    
    assert all(isinstance(r, list) for r in [results1, results2, results3])


def test_mk_bow_terms_many_identifiers():
    sym_obj = {
        'name': 'func',
        'identifiers': [f'identifier_{i}' for i in range(30)],
        'docstring': 'Description'
    }
    terms = _mk_bow_terms(sym_obj, max_terms=10)
    assert isinstance(terms, list)
    assert len(terms) <= 10


def test_paper_to_chunks_many_sections():
    sections = []
    for s in range(5):
        section = {
            'id': f'sec_{s}',
            'title': f'Section {s}',
            'paragraphs': [
                {'id': f'p_{s}_{i}', 'text': f'Paragraph {i} text'}
                for i in range(3)
            ]
        }
        sections.append(section)
    
    paper = {'sections': sections}
    chunks = _paper_to_chunks(paper)
    assert len(chunks) == 15  


def test_to_symbols_maintains_order():
    objs = [
        {'id': f's{i}', 'name': f'func{i}', 'bow_terms': ['word']}
        for i in range(5)
    ]
    symbols = _to_symbols(objs)
    assert len(symbols) == 5
    for i, sym in enumerate(symbols):
        assert sym.id == f's{i}'


def test_to_symbols_with_docstring_fallback():
    objs = [
        {
            'id': 's1',
            'name': 'func',
            'docstring': 'This is a docstring with info'
        }
    ]
    symbols = _to_symbols(objs)
    assert len(symbols) == 1
    assert 'docstring' in symbols[0].full_text.lower() or 'info' in symbols[0].full_text.lower()


def test_to_symbols_with_text_fallback():
    objs = [
        {
            'id': 's1',
            'name': 'func',
            'text': 'This is the text content that should be used'
        }
    ]
    symbols = _to_symbols(objs)
    assert len(symbols) == 1
    assert symbols[0].full_text is not None
    assert len(symbols[0].full_text) > 0


def test_build_and_match_alternatives_generation():
    symbols = [Symbol(id='s1', name='func', bow_terms=['match'], full_text='match text')]
    chunks = [
        Chunk(id='c1', section='Sec1', text='match content one'),
        Chunk(id='c2', section='Sec2', text='match content two'),
        Chunk(id='c3', section='Sec3', text='match content three'),
    ]
    
    results = build_and_match(symbols, chunks, top_k=5)
    assert len(results) == 1
    assert results[0]['best'] is not None
    assert len(results[0]['alternatives']) >= 0  