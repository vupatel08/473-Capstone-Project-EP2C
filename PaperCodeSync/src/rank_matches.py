"""
EP2C – Step 3: Symbol → Paper-Chunk Matching via BM25/TF-IDF + Embeddings
-------------------------------------------------------------------------

What this does
==============
1) Builds a *textual-overlap* index over paper chunks using TF-IDF or BM25.
2) For each symbol query (bag-of-words or short text), retrieves the top-k chunks by overlap.
3) Builds a *semantic* index by encoding queries & chunks with a sentence-embedding model.
4) Computes cosine similarity for semantic matches.
5) Combines the two scores with an adjustable weight to produce final rankings.
6) Saves the top matches (top-1 + runner-ups) for each symbol to a JSONL file.

Inputs
======
- A list of "symbols": [{'id': str, 'name': str, 'bow_terms': List[str], 'full_text': str (optional)}]
- A list of "chunks":  [{'id': str, 'section': str, 'text': str}]
  *OR* a full Paper.json with sections->paragraphs (will auto-flatten).

Outputs
=======
- matches.jsonl: one line per symbol with fields:
  {
    "symbol_id": "...",
    "best": {"chunk_id": "...", "score": float, "overlap_score": float, "semantic_score": float},
    "alternatives": [
        {"chunk_id": "...", "score": float, "overlap_score": float, "semantic_score": float}, ...
    ],
    "topk_raw": [
        {"chunk_id": "...", "combined": float, "overlap": float, "semantic": float}, ...
    ]
  }

Requirements
============
pip install:
  - scikit-learn
  - rank-bm25        (optional; if unavailable, we use TF-IDF only)
  - sentence-transformers (or any embedding backend you prefer)
"""

from __future__ import annotations
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# simple helper to clean up text 
def _normalize(text: str) -> str:
    return (text or "").strip()

# takes a list of words and joins them with spaces 
def _safe_join_terms(terms: List[str]) -> str:
    return " ".join(t for t in terms if t)

# normalizes numbers between 0 and 1 so we can compare different types of scores fairly
def _min_max_norm(values: List[float]) -> List[float]:
    if not values:
        return values
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        return [0.0 for _ in values] 
    return [(v - vmin) / (vmax - vmin) for v in values]

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# splits text into words 
# note: i decided to use regex bc split() wasn't working well
def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in TOKEN_RE.findall(text)]

# class which does the text matching using either BM25 or TF-IDF based on config
class OverlapIndex:
    def __init__(self, method: str = "bm25"):
        method = (method or "bm25").lower()
        self.method = method if method in {"bm25", "tfidf"} else "tfidf"
        self._bm25 = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._tokenized_docs = None
        self._docs = []
        self._ids = []

    def fit(self, chunk_ids: List[str], chunk_texts: List[str]) -> "OverlapIndex":
        self._ids = list(chunk_ids)
        self._docs = [_normalize(t) for t in chunk_texts]

        if self.method == "bm25":
            try:
                from rank_bm25 import BM25Okapi
                self._tokenized_docs = [doc.split() for doc in self._docs]
                self._bm25 = BM25Okapi(self._tokenized_docs)
            except Exception as e:
                print(f"[OverlapIndex] BM25 unavailable ({e}); falling back to TF-IDF.")
                self.method = "tfidf"

        if self.method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(self._docs)

        return self

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        qt = _normalize(query_text)
        if not qt:
            return []
        if self.method == "bm25" and self._bm25 is not None:
            tokens = qt.split()
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
            return [(self._ids[i], float(s)) for i, s in ranked]
        else:
            from sklearn.metrics.pairwise import linear_kernel
            qv = self._tfidf_vectorizer.transform([qt])
            cosine_similarities = linear_kernel(qv, self._tfidf_matrix).flatten()
            ranked_idx = cosine_similarities.argsort()[::-1][:top_k]
            return [(self._ids[i], float(cosine_similarities[i])) for i in ranked_idx]


# uses a simple semantic understanding model to build an index
class SemanticIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._chunk_vecs = None
        self._ids = []

    def fit(self, chunk_ids: List[str], chunk_texts: List[str]) -> "SemanticIndex":
        self._ids = list(chunk_ids)
        texts = [_normalize(t) for t in chunk_texts]
        self._ensure_model()
        self._chunk_vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return self

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        qt = _normalize(query_text)
        if not qt:
            return []
        self._ensure_model()
        qv = self._model.encode([qt], normalize_embeddings=True, show_progress_bar=False)[0]
        import numpy as np
        sims = np.dot(self._chunk_vecs, qv)
        ranked = sims.argsort()[::-1][:top_k]
        return [(self._ids[i], float(sims[i])) for i in ranked]


# stores both the word-matching and semantic scores for a match 
@dataclass
class CombinedMatch:
    chunk_id: str
    combined: float
    overlap: float
    semantic: float

# combines the word-matching scores with the semantic scores 
def combine_scores(
    overlap: List[Tuple[str, float]],
    semantic: List[Tuple[str, float]],
    weight_overlap: float = 0.5,
    weight_semantic: float = 0.5,
    top_k: int = 10
) -> List[CombinedMatch]:
    o_map = {cid: s for cid, s in overlap}
    s_map = {cid: s for cid, s in semantic}
    all_ids = list({*o_map.keys(), *s_map.keys()})
    overlap_scores = [o_map.get(cid, 0.0) for cid in all_ids]
    semantic_scores = [s_map.get(cid, 0.0) for cid in all_ids]

    overlap_norm = _min_max_norm(overlap_scores)
    semantic_norm = _min_max_norm(semantic_scores)

    combined = [
        CombinedMatch(
            chunk_id=cid,
            combined=weight_overlap * o + weight_semantic * s,
            overlap=o,
            semantic=s
        )
        for cid, o, s in zip(all_ids, overlap_norm, semantic_norm)
    ]
    combined.sort(key=lambda m: m.combined, reverse=True)
    return combined[:top_k]

@dataclass
class Symbol:
    id: str
    name: str
    bow_terms: List[str]
    full_text: Optional[str] = None

@dataclass
class Chunk:
    id: str
    section: str
    text: str

# main function that does all the work
def build_and_match(
    symbols: List[Symbol],
    chunks: List[Chunk],
    method_overlap: str = "bm25",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    weight_overlap: float = 0.5,
    weight_semantic: float = 0.5,
    top_k: int = 10
) -> List[Dict]:
    chunk_ids = [c.id for c in chunks]
    chunk_texts = [c.text for c in chunks]

    ovl = OverlapIndex(method=method_overlap).fit(chunk_ids, chunk_texts)
    sem = SemanticIndex(model_name=embed_model).fit(chunk_ids, chunk_texts)

    results = []
    for sym in symbols:
        query_text = _safe_join_terms(sym.bow_terms) or sym.full_text or sym.name
        o_top = ovl.query(query_text, top_k=top_k)
        s_query = sym.full_text or query_text
        s_top = sem.query(s_query, top_k=top_k)

        combined = combine_scores(
            overlap=o_top, semantic=s_top,
            weight_overlap=weight_overlap, weight_semantic=weight_semantic,
            top_k=top_k
        )

        row = {
            "symbol_id": sym.id,
            "symbol_name": sym.name,
            "query_text": query_text,
            "best": None,
            "alternatives": [],
            "topk_raw": [vars(m) for m in combined]
        }
        if combined:
            row["best"] = {
                "chunk_id": combined[0].chunk_id,
                "score": combined[0].combined,
                "overlap_score": combined[0].overlap,
                "semantic_score": combined[0].semantic,
            }
            for m in combined[1:3]:
                row["alternatives"].append({
                    "chunk_id": m.chunk_id,
                    "score": m.combined,
                    "overlap_score": m.overlap,
                    "semantic_score": m.semantic,
                })
        results.append(row)
    return results


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(rows: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# makes a "bag of words" from code symbols 
# note: i had to filter out common words like "the," etc. bc they were messing up the results
def _mk_bow_terms(sym_obj: dict, max_terms: int = 20) -> List[str]:
    name = sym_obj.get("name") or ""
    identifiers = sym_obj.get("identifiers") or []
    doc = sym_obj.get("docstring") or ""
    txt = sym_obj.get("text") or ""

    seeds = []
    for ident in identifiers:
        seeds.extend(_tokenize(ident))
    seeds.extend(_tokenize(name))
    seeds.extend(_tokenize(doc))
    seeds.extend(_tokenize(" ".join(txt.split()[:200])))

    stop = {
        "the","a","an","and","or","if","in","of","to","for","on","with","by","is","are","be","as",
        "this","that","it","we","you","from","at","into","over","under","while","where","which",
        "true","false","none","null","return","class","def","self","args","kwargs"
    }
    toks = [t for t in seeds if len(t) > 2 and t not in stop]
    if not toks:
        toks = _tokenize(name) or ["symbol"]

    counts = Counter(toks)
    return [w for w, _ in counts.most_common(max_terms)]

# converts raw JSON data into the symbol class 
def _to_symbols(objs: List[Dict]) -> List[Symbol]:
    out: List[Symbol] = []
    for o in objs:
        sid = o.get("id") or o.get("name") or "sym:unknown"
        name = o.get("name") or sid

        bow = o.get("bow_terms")
        if not bow:
            bow = _mk_bow_terms(o, max_terms=20)

        ft = o.get("full_text")
        if not ft:
            doc = (o.get("docstring") or "").strip()
            if doc:
                ft = doc
            else:
                ft = " ".join((o.get("text") or "").split()[:120])

        out.append(Symbol(id=sid, name=name, bow_terms=bow, full_text=ft))
    return out

# takes a paper and breaks it into smaller pieces
def _paper_to_chunks(paper: Dict) -> List[Chunk]:
    chunks: List[Chunk] = []
    for sec in paper.get("sections", []):
        sec_title = sec.get("title") or ""
        for p in sec.get("paragraphs", []) or []:
            pid = p.get("id") or f"{sec.get('id','sec')}:p?"
            txt = p.get("text") or ""
            if txt.strip():
                chunks.append(Chunk(id=pid, section=sec_title, text=txt))
    return chunks

# load different types of paper formats
def load_chunks_flexible(path: str) -> List[Chunk]:
    obj = load_json(path)

    if isinstance(obj, list):
        mapped: List[Chunk] = []
        for x in obj:
            if not isinstance(x, dict):
                continue
            cid = x.get("id") or x.get("chunk_id") or "chunk:?"
            sec = x.get("section") or x.get("title") or ""
            txt = x.get("text") or x.get("content") or ""
            if txt:
                mapped.append(Chunk(id=cid, section=sec, text=txt))
        if mapped:
            return mapped

    if isinstance(obj, dict) and "sections" in obj:
        return _paper_to_chunks(obj)

    raise ValueError(f"Unrecognized chunks format in {path!r}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="EP2C Step 3 matcher")
    p.add_argument("--symbols", type=str, default="", help="Path to symbols.json (list of symbols)")
    p.add_argument("--chunks", type=str, default="", help="Path to chunks.json or Paper.json")
    p.add_argument("--overlap", type=str, default="bm25", choices=["bm25", "tfidf"], help="Overlap index type")
    p.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    p.add_argument("--alpha", type=float, default=0.5, help="Weight for textual overlap (0..1). Semantic gets (1-alpha).")
    p.add_argument("--topk", type=int, default=10, help="Top-K to keep")
    p.add_argument("--out", type=str, default="matches.jsonl", help="Output JSONL path")
    args = p.parse_args()

    if args.symbols and os.path.exists(args.symbols):
        raw_syms = load_json(args.symbols)
        if not isinstance(raw_syms, list):
            raise ValueError(f"--symbols must be a JSON list, got {type(raw_syms)}")
        symbols = _to_symbols(raw_syms)

    if args.chunks and os.path.exists(args.chunks):
        chunks = load_chunks_flexible(args.chunks)

    results = build_and_match(
        symbols=symbols,
        chunks=chunks,
        method_overlap=args.overlap,
        embed_model=args.embed_model,
        weight_overlap=args.alpha,
        weight_semantic=1.0 - args.alpha,
        top_k=args.topk
    )

    save_jsonl(results, args.out)
    print(f"[ok] Wrote {len(results)} rows to {args.out}")
