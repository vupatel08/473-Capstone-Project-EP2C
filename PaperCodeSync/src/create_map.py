from __future__ import annotations
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

def _normalize(text: str) -> str:
    return (text or "").strip()

def _safe_join_terms(terms: List[str]) -> str:
    return " ".join(t for t in terms if t)

def _min_max_norm(values: List[float]) -> List[float]:
    if not values:
        return values
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        return [0.0 for _ in values] 
    return [(v - vmin) / (vmax - vmin) for v in values]

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    # note: i decided to use regex bc split() wasn't working well
    return [t.lower() for t in TOKEN_RE.findall(text)]

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


@dataclass
class CombinedMatch:
    chunk_id: str
    combined: float
    overlap: float
    semantic: float

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

    # note: i had to filter out common words like "the," etc. bc they were messing up the results
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

    parser = argparse.ArgumentParser(description="EP2C Step 3 matcher")
    parser.add_argument("--symbols", type=str, default="", help="Path to symbols.json (list of symbols)")
    parser.add_argument("--chunks", type=str, default="", help="Path to chunks.json or Paper.json")
    parser.add_argument("--overlap", type=str, default="bm25", choices=["bm25", "tfidf"], help="Overlap index type")
    parser.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for textual overlap (0..1). Semantic gets (1-alpha).")
    parser.add_argument("--topk", type=int, default=10, help="Top-K to keep")
    parser.add_argument("--out", type=str, default="matches.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    symbols = []
    if args.symbols and os.path.exists(args.symbols):
        raw_syms = load_json(args.symbols)
        if isinstance(raw_syms, list):
            symbols = _to_symbols(raw_syms)
        else:
            print("The symbols file does not contain a list. Please check its format.")
            symbols = []
    else:
        print("[Warning] No symbols file found or path missing.")
        symbols = []

    chunks = []
    if args.chunks and os.path.exists(args.chunks):
        try:
            chunks = load_chunks_flexible(args.chunks)
        except Exception as e:
            print("Error while loading chunks:", e)
            chunks = []
    else:
        print("[Warning] No chunks file found or path missing.")
        chunks = []

    if len(symbols) == 0:
        print("No symbols loaded. Exiting.")
        exit(1)
    if len(chunks) == 0:
        print("No paper chunks loaded. Exiting.")
        exit(1)

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
