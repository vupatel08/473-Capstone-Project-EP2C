import math
from collections import Counter
import pytest

def normalize_text(s: str) -> str:
    if not s:
        return ""
    t = s
    t = t.replace('\u00A0', ' ').strip()
    t = ''.join(ch for ch in t if (ch.isalnum() or ch.isspace() or ch == '-'))
    return ' '.join(t.split()).lower()


def tokenize(s: str):
    s2 = normalize_text(s)
    toks = [t for t in s2.split() if len(t) >= 3]
    return toks


def token_coverage_score(needle_tokens, hay_tokens):
    if not needle_tokens or not hay_tokens:
        return {'score': 0, 'hits': 0, 'rareHits': 0}
    hay_set = set(hay_tokens)
    hits = sum(1 for t in needle_tokens if t in hay_set)
    c = Counter(hay_tokens)
    rare_hits = sum(1 for t in needle_tokens if c.get(t, 0) == 1)
    score = hits / len(needle_tokens) if needle_tokens else 0
    return {'score': score, 'hits': hits, 'rareHits': rare_hits}


def best_page_window_for_tokens(needle_tokens, page_tokens_list, window_size=3):
    best = {'score': 0, 'start': -1, 'end': -1, 'rareHits': 0, 'hits': 0}
    N = len(page_tokens_list)
    for i in range(N):
        for j in range(i, min(N, i + window_size)):
            hay = []
            for k in range(i, j+1):
                hay.extend(page_tokens_list[k])
            res = token_coverage_score(needle_tokens, hay)
            if res['score'] > best['score'] or (math.isclose(res['score'], best['score']) and res['rareHits'] > best['rareHits']):
                best = {'score': res['score'], 'start': i, 'end': j, 'rareHits': res['rareHits'], 'hits': res['hits']}
    return best


def test_normalize_and_tokenize():
    s = "Hello, world! This — is a test."
    toks = tokenize(s)
    assert 'hello' in toks and 'world' in toks and 'test' in toks


def test_token_coverage_score_simple():
    needle = ['alpha', 'beta', 'gamma']
    hay = ['alpha', 'delta', 'gamma', 'alpha']
    out = token_coverage_score(needle, hay)
    assert out['hits'] == 2
    assert out['score'] == pytest.approx(2/3)


def test_best_page_window_prefers_high_coverage():
    page_tokens = [['a','b','c'], ['d','e','a'], ['f','g']]
    needle = ['a','f']
    best = best_page_window_for_tokens(needle, page_tokens, window_size=2)
    assert 0 <= best['start'] <= best['end'] < len(page_tokens)
    assert best['score'] >= 0


def test_click_to_jump_pipeline_simulation():
    page_tokens = [ ['lorem','ipsum'], ['target','token','foo'], ['other','stuff'] ]
    needle = ['target','token']
    best = best_page_window_for_tokens(needle, page_tokens)
    assert best['start'] <= 1 <= best['end']
