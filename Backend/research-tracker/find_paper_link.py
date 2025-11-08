import re
import argparse
from typing import Optional
from pypdf import PdfReader

ARXIV_ID_RE = re.compile(r'\b([0-9]{4}\.[0-9]{4,5})(?:v\d+)?\b', re.I)
ARXIV_URL_RE = re.compile(r'https?://arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?', re.I)
ARXIV_PREFIX_RE = re.compile(r'arxiv:\s*([0-9]{4}\.[0-9]{4,5})(?:v\d+)?', re.I)

def _normalize_arxiv_id(arx: str) -> str:
    base = re.sub(r'v\d+$', '', arx.strip(), flags=re.I)
    return f"https://arxiv.org/abs/{base}"

def _extract_arxiv_from_text(text: str) -> Optional[str]:
    m = ARXIV_URL_RE.search(text)
    if m:
        return _normalize_arxiv_id(m.group(1))
    m = ARXIV_PREFIX_RE.search(text)
    if m:
        return _normalize_arxiv_id(m.group(1))
    m = ARXIV_ID_RE.search(text)
    if m:
        return _normalize_arxiv_id(m.group(1))
    return None

def _pdf_to_arxiv(pdf_path: str, max_pages: int = 2) -> Optional[str]:
    reader = PdfReader(pdf_path)
    pages = min(max_pages, len(reader.pages))
    buf = []
    for i in range(pages):
        try:
            buf.append(reader.pages[i].extract_text() or "")
        except Exception:
            pass
    text = "\n".join(buf)
    return _extract_arxiv_from_text(text)


def find_link(paper_path: str) -> Optional[str]:
    return _pdf_to_arxiv(paper_path)