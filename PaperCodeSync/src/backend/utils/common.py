import hashlib, json, re, os
from pathlib import Path
from typing import Iterable, List, Any, Optional
from utils.parse_config import load_config

config = load_config("../config.yaml")
utils  = config.get("utils", {})

SLUGIFY_MAXLEN = int(utils.get("slugify_maxlen", 80))
ID_TRUNCATE    = int(utils.get("id_truncate", 12))
_TOKEN_RE      = re.compile(utils.get("token_pattern", r"[A-Za-z0-9_]+"))

_TEXT          = utils.get("text", {})
PARA_MIN_LEN   = int(_TEXT.get("min_paragraph_len", 20))
CHUNK_MAX      = int(_TEXT.get("chunk_max_chars", 1400))
CHUNK_HARD_MAX = int(_TEXT.get("chunk_hard_max_chars", max(CHUNK_MAX, 1600)))
PARA_JOIN      = str(_TEXT.get("paragraph_join", "\n\n"))

def slugify(text: str, maxlen: int = SLUGIFY_MAXLEN) -> str:
    if not text:
        return ""
    s = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return (s or "untitled")[:maxlen]

def sha1_prefix(s: str, n: int = ID_TRUNCATE) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

def keep_text(s: str, min_len: Optional[int] = None) -> bool:
    if min_len is None:
        min_len = PARA_MIN_LEN
    return bool(s and len(s.strip()) >= int(min_len))

def split_into_chunks(
    text: str,
    max_chars: Optional[int] = None,
    joiner: Optional[str] = None,
    hard_max: Optional[int] = None,
) -> List[str]:
    if max_chars is None:
        max_chars = CHUNK_MAX
    if hard_max is None:
        hard_max = CHUNK_HARD_MAX
    if joiner is None:
        joiner = PARA_JOIN

    out: List[str] = []
    cur: List[str] = []
    cur_len = 0
    jlen = len(joiner)

    def flush_cur():
        nonlocal cur, cur_len
        if cur:
            out.append(joiner.join(cur))
            cur, cur_len = [], 0

    for block in text.split(joiner):
        if len(block) > hard_max:
            flush_cur()
            start = 0
            while start < len(block):
                out.append(block[start : start + hard_max])
                start += hard_max
            continue

        projected = cur_len + (jlen if cur else 0) + len(block)
        if cur and projected > max_chars:
            flush_cur()
        cur.append(block)
        cur_len = (cur_len + (jlen if cur_len else 0) + len(block)) if cur_len else len(block)

    flush_cur()
    return out

def sha1_hex(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def sha1_id(*parts: str, prefix: str = "") -> str:
    h = hashlib.sha1()
    for p in parts:
        if p:
            h.update(p.encode("utf-8")); h.update(b"\x00")
    base = h.hexdigest()[:ID_TRUNCATE]
    return f"{prefix}{base}" if prefix else base

def uniq_sorted(xs: Iterable[str]) -> List[str]:
    return sorted(set(xs))

def normalize_text(text: str) -> str:
    return (text or "").strip()

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]

def safe_join_terms(terms: Iterable[str]) -> str:
    return " ".join(t for t in terms if t)

def min_max_norm(values: List[float]) -> List[float]:
    if not values:
        return values
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    rng = (vmax - vmin)
    return [(v - vmin) / rng for v in values]

def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(rows: Iterable[Any], path: str | os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_text_safe(path: str | os.PathLike) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def read_bytes_safe(path: str | os.PathLike) -> bytes:
    try:
        return Path(path).read_bytes()
    except Exception:
        return b""

def posix_path(p: str | os.PathLike) -> str:
    return Path(p).as_posix()

def slice_text(src: bytes, node) -> bytes:
    return src[node.start_byte: node.end_byte]
