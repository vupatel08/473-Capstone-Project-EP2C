import hashlib, json, re, os
from pathlib import Path
from typing import Iterable, List, Any
from utils.parse_config import load_config

config = load_config("../../config.yaml")
utils = config['utils']

SLUGIFY_MAXLEN = utils["slugify_maxlen"]
ID_TRUNCATE = utils["id_truncate"]
_TOKEN_RE = re.compile(utils["token_pattern"])

def slugify(text: str, maxlen: int = SLUGIFY_MAXLEN) -> str:
    if not text:
        return ""
    s = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return (s or "untitled")[:maxlen]

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
