import re

# patterns we can reuse
HEADING_RE     = re.compile(r'^(#{1,6})\s+(.*)\s*$')
IMG_RE         = re.compile(r'!\[(.*?)\]\((.*?)\)')
FENCED_EQ_RE   = re.compile(r'^\s*\${2}\s*$')  
