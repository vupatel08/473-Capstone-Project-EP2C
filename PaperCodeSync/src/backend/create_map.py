# """
# EP2C – Step 3: Symbol → Paper-Chunk Matching via BM25/TF-IDF + Embeddings
# -------------------------------------------------------------------------

# What this does
# ==============
# 1) Builds a *textual-overlap* index over paper chunks using TF-IDF or BM25.
# 2) For each symbol query (bag-of-words or short text), retrieves the top-k chunks by overlap.
# 3) Builds a *semantic* index by encoding queries & chunks with a sentence-embedding model.
# 4) Computes cosine similarity for semantic matches.
# 5) Combines the two scores with an adjustable weight to produce final rankings.
# 6) Saves the top matches (top-1 + runner-ups) for each symbol to a JSONL file.

# Inputs
# ======
# - A list of "symbols": [{'id': str, 'name': str, 'bow_terms': List[str], 'full_text': str (optional)}]
# - A list of "chunks":  [{'id': str, 'section': str, 'text': str}]
#   *OR* a full Paper.json with sections->paragraphs (will auto-flatten).

# Outputs
# =======
# - matches.jsonl: one line per symbol with fields:
#   {
#     "symbol_id": "...",
#     "best": {"chunk_id": "...", "score": float, "overlap_score": float, "semantic_score": float},
#     "alternatives": [
#         {"chunk_id": "...", "score": float, "overlap_score": float, "semantic_score": float}, ...
#     ],
#     "topk_raw": [
#         {"chunk_id": "...", "combined": float, "overlap": float, "semantic": float}, ...
#     ]
#   }

# Requirements
# ============
# pip install:
#   - scikit-learn
#   - rank-bm25        (optional; if unavailable, we use TF-IDF only)
#   - sentence-transformers (or any embedding backend you prefer)
# """
import os
import json
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from utils.common import normalize_text, safe_join_terms, min_max_norm, tokenize, load_json, save_jsonl
from utils.parse_config import load_config

config = load_config("../config.yaml")
map_ = config["map"]

OVERLAP_METHOD = map_["overlap_method"]               
ALPHA = map_["alpha"]                                
TOP_K = map_["top_k"]
NORMALIZATION = map_["normalization"]                 

BM25_K1 = map_["bm25"]["k1"]
BM25_B = map_["bm25"]["b"]

TFIDF_MIN_N = map_["tfidf"]["ngram_min"]
TFIDF_MAX_N = map_["tfidf"]["ngram_max"]
TFIDF_STOP = map_["tfidf"]["stop_words"] or None
TFIDF_SUBLINEAR = map_["tfidf"]["sublinear_tf"]
TFIDF_USE_IDF = map_["tfidf"]["use_idf"]

SEM_MODEL = map_["semantic_model"]
SEM_NORMALIZE = map_["semantic"]["normalize_embeddings"]
SEM_BATCH = map_["semantic"]["batch_size"]

Q_USE_FT_FOR_SEM = map_["query"]["use_full_text_for_semantic"]
Q_USE_FT_FOR_OVL = map_["query"]["use_full_text_for_overlap"]
Q_BOW_MAX = map_["query"]["bow_max_terms"]


class OverlapIndex:
    def __init__(self, method: str = OVERLAP_METHOD):
        m = (method or "").lower()
        if m not in {"bm25", "tfidf"}:
            raise ValueError(f"Invalid overlap method: {method}")
        self.method = m
        self._bm25 = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._docs = []
        self._ids = []

    def fit(self, chunk_ids: List[str], chunk_texts: List[str]) -> "OverlapIndex":
        self._ids = list(chunk_ids)
        self._docs = [normalize_text(t) for t in chunk_texts]

        if self.method == "bm25":
            from rank_bm25 import BM25Okapi  
            tokenized = [doc.split() for doc in self._docs]
            self._bm25 = BM25Okapi(tokenized, k1=BM25_K1, b=BM25_B)

        elif self.method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(
                stop_words=TFIDF_STOP,
                ngram_range=(TFIDF_MIN_N, TFIDF_MAX_N),
                sublinear_tf=TFIDF_SUBLINEAR,
                use_idf=TFIDF_USE_IDF,
            )
            self._tfidf_vectorizer = vec
            self._tfidf_matrix = vec.fit_transform(self._docs)

        return self

    def query(self, query_text: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        qt = normalize_text(query_text)
        if not qt:
            return []

        if self.method == "bm25":
            tokens = qt.split()
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
            return [(self._ids[i], float(s)) for i, s in ranked]

        from sklearn.metrics.pairwise import linear_kernel
        qv = self._tfidf_vectorizer.transform([qt])
        sims = linear_kernel(qv, self._tfidf_matrix).flatten()
        ranked_idx = sims.argsort()[::-1][:top_k]
        return [(self._ids[i], float(sims[i])) for i in ranked_idx]


class SemanticIndex:
    def __init__(self, model_name: str = SEM_MODEL):
        self.model_name = model_name
        self._model = None
        self._chunk_vecs = None
        self._ids = []

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def fit(self, chunk_ids: List[str], chunk_texts: List[str]) -> "SemanticIndex":
        self._ids = list(chunk_ids)
        texts = [normalize_text(t) for t in chunk_texts]
        self._ensure_model()
        self._chunk_vecs = self._model.encode(
            texts,
            batch_size=SEM_BATCH,
            normalize_embeddings=SEM_NORMALIZE,
            show_progress_bar=False,
        )
        return self

    def query(self, query_text: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        qt = normalize_text(query_text)
        if not qt:
            return []
        self._ensure_model()
        qv = self._model.encode(
            [qt],
            batch_size=1,
            normalize_embeddings=SEM_NORMALIZE,
            show_progress_bar=False,
        )[0]
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


def _maybe_normalize(overlap_scores: List[float], semantic_scores: List[float]):
    if NORMALIZATION == "none":
        return overlap_scores, semantic_scores
    return min_max_norm(overlap_scores), min_max_norm(semantic_scores)


def combine_scores(
    overlap: List[Tuple[str, float]],
    semantic: List[Tuple[str, float]],
    weight_overlap: float = ALPHA,
    weight_semantic: float = 1.0 - ALPHA,
    top_k: int = TOP_K
) -> List[CombinedMatch]:
    o_map = {cid: s for cid, s in overlap}
    s_map = {cid: s for cid, s in semantic}
    all_ids = list({*o_map.keys(), *s_map.keys()})
    overlap_scores = [o_map.get(cid, 0.0) for cid in all_ids]
    semantic_scores = [s_map.get(cid, 0.0) for cid in all_ids]

    o_n, s_n = _maybe_normalize(overlap_scores, semantic_scores)

    merged = [
        CombinedMatch(
            chunk_id=cid,
            combined=weight_overlap * o + weight_semantic * s,
            overlap=o,
            semantic=s
        )
        for cid, o, s in zip(all_ids, o_n, s_n)
    ]
    merged.sort(key=lambda m: m.combined, reverse=True)
    return merged[:top_k]


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


def _mk_bow_terms(sym_obj: dict, max_terms: int = Q_BOW_MAX) -> List[str]:
    name = sym_obj.get("name") or ""
    identifiers = sym_obj.get("identifiers") or []
    doc = sym_obj.get("docstring") or ""
    txt = sym_obj.get("text") or ""

    seeds: List[str] = []
    for ident in identifiers:
        seeds.extend(tokenize(ident))
    seeds.extend(tokenize(name))
    seeds.extend(tokenize(doc))
    seeds.extend(tokenize(" ".join(txt.split()[:200])))

    stop = {
        "the","a","an","and","or","if","in","of","to","for","on","with","by","is","are","be","as",
        "this","that","it","we","you","from","at","into","over","under","while","where","which",
        "true","false","none","null","return","class","def","self","args","kwargs"
    }
    toks = [t for t in seeds if len(t) > 2 and t not in stop]
    if not toks:
        toks = tokenize(name) or ["symbol"]

    counts = Counter(toks)
    return [w for w, _ in counts.most_common(max_terms)]


def _to_symbols(objs: List[Dict]) -> List[Symbol]:
    out: List[Symbol] = []
    for o in objs:
        sid = o.get("id") or o.get("name") or "sym:unknown"
        name = o.get("name") or sid
        bow = o.get("bow_terms") or _mk_bow_terms(o, max_terms=Q_BOW_MAX)
        ft = o.get("full_text")
        if not ft:
            doc = (o.get("docstring") or "").strip()
            ft = doc if doc else " ".join((o.get("text") or "").split()[:120])
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


def build_and_match(
    symbols: List[Symbol],
    chunks: List[Chunk],
    method_overlap: str = OVERLAP_METHOD,
    embed_model: str = SEM_MODEL,
    weight_overlap: float = ALPHA,
    weight_semantic: float = 1.0 - ALPHA,
    top_k: int = TOP_K
) -> List[Dict]:
    chunk_ids = [c.id for c in chunks]
    chunk_texts = [c.text for c in chunks]

    ovl = OverlapIndex(method=method_overlap).fit(chunk_ids, chunk_texts)
    sem = SemanticIndex(model_name=embed_model).fit(chunk_ids, chunk_texts)

    rows = []
    for sym in symbols:
        ovl_query = (sym.full_text if Q_USE_FT_FOR_OVL and sym.full_text
                     else safe_join_terms(sym.bow_terms) or sym.name)
        sem_query = (sym.full_text if Q_USE_FT_FOR_SEM and sym.full_text
                     else safe_join_terms(sym.bow_terms) or sym.name)

        o_top = ovl.query(ovl_query, top_k=top_k)
        s_top = sem.query(sem_query, top_k=top_k)

        combined = combine_scores(
            overlap=o_top,
            semantic=s_top,
            weight_overlap=weight_overlap,
            weight_semantic=weight_semantic,
            top_k=top_k
        )

        row = {
            "symbol_id": sym.id,
            "symbol_name": sym.name,
            "query_text_overlap": ovl_query,
            "query_text_semantic": sem_query,
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
        rows.append(row)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="EP2C – Step 3 matcher (config-driven)")
    ap.add_argument("symbols", type=str, help="Path to symbols.json (list of symbols)")
    ap.add_argument("chunks", type=str, help="Path to chunks.json or Paper.json")
    args = ap.parse_args()

    if not (os.path.exists(args.symbols) and os.path.isfile(args.symbols)):
        raise SystemExit(f"Symbols file not found: {args.symbols}")
    if not (os.path.exists(args.chunks) and os.path.isfile(args.chunks)):
        raise SystemExit(f"Chunks file not found: {args.chunks}")

    raw_syms = load_json(args.symbols)
    if not isinstance(raw_syms, list):
        raise SystemExit("Symbols JSON must be a list of symbol objects.")
    symbols = _to_symbols(raw_syms)

    try:
        chunks = load_chunks_flexible(args.chunks)
    except Exception as e:
        raise SystemExit(f"Failed to load chunks: {e}")

    if not symbols:
        raise SystemExit("No symbols loaded. Exiting.")
    if not chunks:
        raise SystemExit("No paper chunks loaded. Exiting.")

    results = build_and_match(
        symbols=symbols,
        chunks=chunks,
        method_overlap=OVERLAP_METHOD,
        embed_model=SEM_MODEL,
        weight_overlap=ALPHA,
        weight_semantic=1.0 - ALPHA,
        top_k=TOP_K
    )

    out_path = "matches.jsonl"
    save_jsonl(results, out_path)
    print(f"[ok] Wrote {len(results)} rows to {out_path}")
