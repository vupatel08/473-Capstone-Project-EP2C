import re

def leading_hash_comments_python(src: bytes, node) -> str:
    start_row = node.start_point[0]
    text = src.decode("utf-8", "ignore")
    lines = text.splitlines()
    i = start_row - 1
    buf = []
    while i >= 0:
        line = lines[i]
        if not line.strip():
            if buf: break
            i -= 1; continue
        if re.match(r"^\s*#", line):
            buf.append(re.sub(r"^\s*#\s?", "", line)); i -= 1; continue
        break
    return "\n".join(reversed(buf)).strip()

def leading_docblock_or_slashes(src: bytes, node) -> str:
    pre = src[: node.start_byte].decode("utf-8", "ignore")
    m = re.search(r"/\*\*[\s\S]*?\*/\s*$", pre)
    if m:
        inner = re.sub(r"^/\*\*|\*/$", "", m.group(0), flags=re.DOTALL).strip()
        inner = re.sub(r"^[ \t]*\* ?", "", inner, flags=re.MULTILINE)
        return inner.strip()
    m2 = re.search(r"(?:^[ \t]*/{2,3}.*\n?)+\s*$", pre, flags=re.MULTILINE)
    if m2:
        lines = [re.sub(r"^[ \t]*/{2,3} ?", "", ln) for ln in m2.group(0).splitlines() if ln.strip()]
        return "\n".join(lines).strip()
    return ""
