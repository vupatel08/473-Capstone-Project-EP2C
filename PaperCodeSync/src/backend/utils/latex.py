import re
from utils.parse_config import load_config

# --- config wiring ---
_config = load_config("../config.yaml")
_utils  = _config.get("utils", {})
_latex     = _utils.get("latex", {})

STRIP_SPACE_AFTER_CMDS   = bool(_latex.get("strip_space_after_commands", True))
COLLAPSE_MATHRM_SPACES   = bool(_latex.get("collapse_mathrm_inner_spaces", True))
COLLAPSE_MULTI_SPACES    = bool(_latex.get("collapse_multi_spaces", True))

LATEX_CMD_SP_GAP   = re.compile(r'\\([A-Za-z]+)\s+')
LATEX_MATHRM_SPANS = re.compile(r'\\mathrm\{\s*([A-Za-z](?:\s+[A-Za-z])+\s*)\}')
LATEX_MULTI_SPACE  = re.compile(r'[ \t]{2,}')

# normalize common LaTeX quirks 
def normalize_latex(s: str) -> str:
    if not s:
        return s
    if STRIP_SPACE_AFTER_CMDS:
        s = LATEX_CMD_SP_GAP.sub(lambda m: f"\\{m.group(1)}", s)
    if COLLAPSE_MATHRM_SPACES:
        s = LATEX_MATHRM_SPANS.sub(lambda m: r'\mathrm{' + re.sub(r'\s+', '', m.group(1)) + '}', s)
    if COLLAPSE_MULTI_SPACES:
        s = LATEX_MULTI_SPACE.sub(' ', s)
    return s.strip()
