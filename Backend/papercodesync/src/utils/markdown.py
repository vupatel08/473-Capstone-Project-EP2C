import re
from utils.parse_config import load_config

_config = load_config("../config.yaml")
_utils  = _config.get("utils", {})
_markdown     = _utils.get("markdown", {})

EQ_FENCE          = str(_markdown.get("eq_fence", "$$")).strip()
REF_HEADINGS_LIST = [str(h).lower() for h in _markdown.get("reference_headings", ["references"])]
FOLD_IMAGE_ALT    = bool(_markdown.get("fold_image_alt_into_prose", True))

# patterns we can reuse
HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)\s*$')
IMG_RE     = re.compile(r'!\[(.*?)\]\((.*?)\)')

FENCED_EQ_RE = re.compile(r'^\s*\${2}\s*$')

def is_eq_fence(line: str) -> bool:
    return line.strip() == EQ_FENCE

def is_reference_heading(text: str | None) -> bool:
    return bool(text) and text.strip().lower() in REF_HEADINGS_LIST
