import re

LATEX_CMD_SP_GAP   = re.compile(r'\\([A-Za-z]+)\s+')
LATEX_MATHRM_SPANS = re.compile(r'\\mathrm\{\s*([A-Za-z](?:\s+[A-Za-z])+)\s*\}')
LATEX_MULTI_SPACE  = re.compile(r'[ \t]{2,}')

# normalize common LaTeX quirks
def normalize_latex(s: str) -> str:
    if not s:
        return s
    s = LATEX_CMD_SP_GAP.sub(lambda m: f"\\{m.group(1)}", s)
    s = LATEX_MATHRM_SPANS.sub(lambda m: r'\mathrm{' + m.group(1).replace(' ', '') + '}', s)
    s = LATEX_MULTI_SPACE.sub(' ', s)
    return s.strip()
