import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
from tree_sitter_languages import get_parser
from utils.languages import EXT_TO_LANG
from utils.common import sha1_hex, uniq_sorted, slice_text, read_bytes_safe, posix_path
from utils.comments import leading_hash_comments_python, leading_docblock_or_slashes
from utils.parse_config import load_config

config = load_config("../config.yaml")
symbols = config["symbols"]  

EXCLUDE_DIRS = set(symbols["exclude_dirs"])
FILE_EXCLUDE_RE = re.compile(symbols["file_exclude_regex"]) if symbols.get("file_exclude_regex") else None

KEEP_TEXT_SPAN = symbols["keep_text_span"]
TEXT_MAX_CHARS = symbols["text_max_chars"]
MAX_FILE_BYTES = symbols["max_file_bytes"]

INCLUDE_SYMBOL_KINDS = set(k.lower() for k in symbols["include_symbol_kinds"])

PYTHON_DOC_MERGE_STRATEGY = symbols["python_doc_merge_strategy"]  

EMIT_MODULE_SPAN_DEFAULT = symbols["emit_module_span_default"]
MODULE_HEADER_CAPTURE = symbols["module_header_capture"]
MODULE_HEADER_MAX_LINES = symbols["module_header_max_lines"]

FOLLOW_SYMLINKS = symbols["follow_symlinks"]

PARSER_MAP = {
    "python": get_parser("python"),
    "javascript": get_parser("javascript"),
    "typescript": get_parser("typescript"),
    "tsx": get_parser("tsx"),
    "java": get_parser("java"),
    "cpp": get_parser("cpp"),
}

@dataclass
class SymbolRec:
    id: str
    file: str
    kind: str
    name: str
    signature: str
    docstring: str
    identifiers: List[str]
    start_line: int
    end_line: int
    text: str


def detect_language(path: Path) -> Optional[str]:
    lang = EXT_TO_LANG.get(path.suffix.lower())
    return lang if lang in PARSER_MAP else None


def make_id(file: str, kind: str, name: str, start_line: int, end_line: int, text: bytes) -> str:
    key = f"{file}|{kind}|{name}|{start_line}|{end_line}|".encode() + text
    return sha1_hex(key)

def _merge_python_docs(py_doc: str, lead: str) -> str:
    if PYTHON_DOC_MERGE_STRATEGY == "only_docstring":
        return (py_doc or "").strip()
    if PYTHON_DOC_MERGE_STRATEGY == "only_leading":
        return (lead or "").strip()
    return (py_doc + ("\n" if py_doc and lead else "") + lead).strip()

def _maybe_truncate_text(s: str) -> str:
    if not KEEP_TEXT_SPAN:
        return ""
    if TEXT_MAX_CHARS and len(s) > TEXT_MAX_CHARS:
        return s[:TEXT_MAX_CHARS]
    return s

def _path_blocked(path: Path) -> bool:
    if FILE_EXCLUDE_RE and FILE_EXCLUDE_RE.search(path.as_posix()):
        return True
    return False

def python_extract_symbols(src: bytes, tree) -> List[SymbolRec]:
    root = tree.root_node
    recs: List[SymbolRec] = []

    def node_text(n) -> str:
        return slice_text(src, n).decode("utf-8", "ignore")

    def first_docstring(n) -> str:
        for child in n.children:
            if child.type in {"block", "suite"} and child.children:
                first = child.children[0]
                if first.type == "expression_statement" and first.children and \
                   first.children[0].type in {"string", "string_literal"}:
                    return node_text(first.children[0]).strip("'\" ")
        return ""

    def collect_identifiers(n) -> List[str]:
        ids = []
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur.type == "identifier":
                ids.append(node_text(cur).lower())
            stack.extend(cur.children)
        return uniq_sorted(ids)

    def parms_text(n) -> str:
        for ch in n.children:
            if ch.type in {"parameters", "lambda_parameters"}:
                return node_text(ch)
        return ""

    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in {"function_definition", "class_definition"}:
            kind = "function" if n.type == "function_definition" else "class"
            if kind not in INCLUDE_SYMBOL_KINDS:
                stack.extend(n.children); continue

            name_node = next((ch for ch in n.children if ch.type == "identifier"), None)
            name = node_text(name_node) if name_node else "<anonymous>"

            if kind == "function":
                pt = parms_text(n)
                signature = f"{name}{pt}" if pt else name
            else:
                bases = ""
                for ch in n.children:
                    if ch.type == "argument_list":
                        bases = node_text(ch); break
                signature = f"class {name}{bases}" if bases else f"class {name}"

            py_doc = first_docstring(n)
            lead = leading_hash_comments_python(src, n)
            doc_combined = _merge_python_docs(py_doc, lead)

            ids = collect_identifiers(n)
            start_line = n.start_point[0] + 1
            end_line = n.end_point[0] + 1
            span_bytes = slice_text(src, n)
            span_text = _maybe_truncate_text(span_bytes.decode("utf-8", "ignore"))

            sid = make_id("<FILE>", kind, name, start_line, end_line, span_bytes)

            recs.append(SymbolRec(
                id=sid,
                file="",
                kind=kind,
                name=name,
                signature=signature,
                docstring=doc_combined,
                identifiers=ids,
                start_line=start_line,
                end_line=end_line,
                text=span_text,
            ))
        stack.extend(n.children)
    return recs


JS_FUNC_TYPES = {
    "function_declaration": "function",
    "method_definition": "method",
    "class_declaration": "class",
}
TS_EXTRA_FUNC_TYPES = {"function_signature": "function"}
IDENTIFIER_TYPES_JS = {"identifier", "property_identifier", "shorthand_property_identifier"}

def js_like_extract_symbols(src: bytes, tree) -> List[SymbolRec]:
    root = tree.root_node
    recs: List[SymbolRec] = []

    def node_text(n) -> str:
        return slice_text(src, n).decode("utf-8", "ignore")

    def collect_identifiers(n) -> List[str]:
        ids = []
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur.type in IDENTIFIER_TYPES_JS:
                ids.append(node_text(cur).lower())
            stack.extend(cur.children)
        return uniq_sorted(ids)

    def name_for(n) -> str:
        for ch in n.children:
            if ch.type in IDENTIFIER_TYPES_JS:
                return node_text(ch)
        return "<anonymous>"

    def params_text(n) -> str:
        for ch in n.children:
            if ch.type in {"formal_parameters", "arguments", "type_parameters"}:
                return node_text(ch)
        return ""

    stack = [root]
    while stack:
        n = stack.pop()
        kind = None
        if n.type in JS_FUNC_TYPES:
            kind = JS_FUNC_TYPES[n.type]
        elif n.type in TS_EXTRA_FUNC_TYPES:
            kind = TS_EXTRA_FUNC_TYPES[n.type]

        if kind and kind in INCLUDE_SYMBOL_KINDS:
            name = name_for(n)
            pt = params_text(n)
            signature = f"{name}{pt}" if pt else name

            ids = collect_identifiers(n)
            start_line = n.start_point[0] + 1
            end_line = n.end_point[0] + 1
            span_bytes = slice_text(src, n)
            span_text = _maybe_truncate_text(span_bytes.decode("utf-8", "ignore"))

            doc_or_jsdoc = leading_docblock_or_slashes(src, n)

            recs.append(SymbolRec(
                id="",
                file="",
                kind=kind,
                name=name,
                signature=signature,
                docstring=doc_or_jsdoc,
                identifiers=ids,
                start_line=start_line,
                end_line=end_line,
                text=span_text,
            ))
        stack.extend(n.children)
    return recs


IDENTIFIER_TYPES_JAVA = {"identifier", "type_identifier"}

def java_extract_symbols(src: bytes, tree) -> List[SymbolRec]:
    root = tree.root_node
    recs: List[SymbolRec] = []

    def node_text(n) -> str:
        return slice_text(src, n).decode("utf-8", "ignore")

    def collect_identifiers(n) -> List[str]:
        ids = []
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur.type in IDENTIFIER_TYPES_JAVA:
                ids.append(node_text(cur).lower())
            stack.extend(cur.children)
        return uniq_sorted(ids)

    def name_for(n) -> str:
        for ch in n.children:
            if ch.type in IDENTIFIER_TYPES_JAVA:
                return node_text(ch)
        return "<anonymous>"

    def params_text(n) -> str:
        for ch in n.children:
            if ch.type in {"formal_parameters"}:
                return node_text(ch)
        return ""

    TARGETS = {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "method_declaration": "method",
        "constructor_declaration": "constructor",
    }

    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in TARGETS:
            kind = TARGETS[n.type]
            if kind not in INCLUDE_SYMBOL_KINDS:
                stack.extend(n.children); continue

            name = name_for(n)
            pt = params_text(n)
            signature = f"{name}{pt}" if pt else name

            ids = collect_identifiers(n)
            start_line = n.start_point[0] + 1
            end_line = n.end_point[0] + 1
            span_bytes = slice_text(src, n)
            span_text = _maybe_truncate_text(span_bytes.decode("utf-8", "ignore"))

            jdoc = leading_docblock_or_slashes(src, n)

            recs.append(SymbolRec(
                id="",
                file="",
                kind=kind,
                name=name,
                signature=signature,
                docstring=jdoc,
                identifiers=ids,
                start_line=start_line,
                end_line=end_line,
                text=span_text,
            ))
        stack.extend(n.children)
    return recs


IDENTIFIER_TYPES_CPP = {"identifier", "type_identifier", "field_identifier"}

def cpp_extract_symbols(src: bytes, tree) -> List[SymbolRec]:
    root = tree.root_node
    recs: List[SymbolRec] = []

    def node_text(n) -> str:
        return slice_text(src, n).decode("utf-8", "ignore")

    def collect_identifiers(n) -> List[str]:
        ids = []
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur.type in IDENTIFIER_TYPES_CPP:
                ids.append(node_text(cur).lower())
            stack.extend(cur.children)
        return uniq_sorted(ids)

    def function_like(n) -> bool:
        if n.type == "function_definition":
            return True
        if n.type == "declaration":
            return any(ch.type == "function_declarator" for ch in n.children)
        return False

    def name_and_sig_for_func(n) -> tuple[str, str]:
        def find_ident(m):
            stack = [m]
            while stack:
                cur = stack.pop()
                if cur.type in {"identifier"}:
                    return node_text(cur)
                stack.extend(cur.children)
            return "<anonymous>"

        if n.type == "function_definition":
            decl = next((ch for ch in n.children if ch.type.endswith("declarator")), None)
            if decl:
                name = find_ident(decl)
                sig = node_text(decl)
                return name, sig

        decl = next((ch for ch in n.children if ch.type == "function_declarator"), None)
        if decl:
            name = find_ident(decl)
            sig = node_text(decl)
            return name, sig

        return "<anonymous>", node_text(n)

    TARGETS = {"class_specifier": "class", "struct_specifier": "struct"}

    stack = [root]
    while stack:
        n = stack.pop()
        kind = None
        name = ""
        signature = ""

        if n.type in TARGETS:
            kind = TARGETS[n.type]
            if kind not in INCLUDE_SYMBOL_KINDS:
                stack.extend(n.children); continue
            name_node = next((ch for ch in n.children if ch.type == "type_identifier"), None)
            name = node_text(name_node) if name_node else "<anonymous>"
            signature = f"{kind} {name}"

        elif function_like(n):
            kind = "function"
            if kind not in INCLUDE_SYMBOL_KINDS:
                stack.extend(n.children); continue
            name, signature = name_and_sig_for_func(n)

        if kind:
            ids = collect_identifiers(n)
            start_line = n.start_point[0] + 1
            end_line = n.end_point[0] + 1
            span_bytes = slice_text(src, n)
            span_text = _maybe_truncate_text(span_bytes.decode("utf-8", "ignore"))

            cdoc = leading_docblock_or_slashes(src, n)

            recs.append(SymbolRec(
                id="",
                file="",
                kind=kind,
                name=name,
                signature=signature,
                docstring=cdoc,
                identifiers=ids,
                start_line=start_line,
                end_line=end_line,
                text=span_text,
            ))
        stack.extend(n.children)
    return recs

def parse_file(path: Path) -> List[SymbolRec]:
    if _path_blocked(path):
        return []

    lang_key = detect_language(path)
    if not lang_key:
        return []

    parser = PARSER_MAP[lang_key]

    src = read_bytes_safe(path)
    if not src:
        return []
    if MAX_FILE_BYTES and len(src) > MAX_FILE_BYTES:
        return []

    tree = parser.parse(src)

    if lang_key == "python":
        recs = python_extract_symbols(src, tree)
    elif lang_key in {"javascript", "typescript", "tsx"}:
        recs = js_like_extract_symbols(src, tree)
    elif lang_key == "java":
        recs = java_extract_symbols(src, tree)
    elif lang_key == "cpp":
        recs = cpp_extract_symbols(src, tree)
    else:
        recs = []

    for r in recs:
        r.file = posix_path(path)
        span_bytes_for_hash = slice_text(src, tree.root_node)[:0]  # no-op placeholder
        span_bytes_for_hash = r.text.encode("utf-8") if KEEP_TEXT_SPAN else b""
        r.id = make_id(r.file, r.kind, r.name, r.start_line, r.end_line, span_bytes_for_hash)
    return recs


def crawl_repo(root: Path) -> List[SymbolRec]:
    symbols: List[SymbolRec] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=FOLLOW_SYMLINKS):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if detect_language(p):
                symbols.extend(parse_file(p))
    return symbols


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EP2C – Step 2 (Tree-sitter) repo symbol extractor")
    ap.add_argument("root", type=str, help="Path to repository root")
    ap.add_argument("--out", type=str, default="symbols.json", help="Output JSON path")
    ap.add_argument("--emit-module-span", action="store_true", help="Also emit a file-level 'module' span per file")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"error: {root} is not a directory")

    symbols = crawl_repo(root)

    if args.emit_module_span or EMIT_MODULE_SPAN_DEFAULT:
        by_file: Dict[str, List[SymbolRec]] = {}
        for s in symbols:
            by_file.setdefault(s.file, []).append(s)

        for file, _ in by_file.items():
            try:
                src = Path(file).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            start = 1
            end = src.count("\n") + 1
            sid = sha1_hex((file + "|module").encode())

            header = ""
            if MODULE_HEADER_CAPTURE:
                m = re.match(r"^(?:\s*(?:#|//).*\n|/\*[\s\S]*?\*/\s*\n)+", src)
                if m:
                    block = m.group(0)
                    block = re.sub(r"^\s*#\s?", "", block, flags=re.MULTILINE)
                    block = re.sub(r"^\s*//\s?", "", block, flags=re.MULTILINE)
                    block = re.sub(r"/\*|\*/", "", block)
                    header = block.strip()
                    if MODULE_HEADER_MAX_LINES:
                        header = "\n".join(header.splitlines()[:MODULE_HEADER_MAX_LINES]).strip()

            text_payload = _maybe_truncate_text(src)

            symbols.append(SymbolRec(
                id=sid,
                file=file,
                kind="module",
                name=Path(file).name,
                signature=Path(file).name,
                docstring=header,
                identifiers=[],
                start_line=start,
                end_line=end,
                text=text_payload,
            ))

    out = [asdict(s) for s in symbols]
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out} with {len(symbols)} symbols")