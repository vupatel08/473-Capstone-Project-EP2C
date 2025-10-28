import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from tree_sitter_languages import get_parser

EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",   
    ".mjs": "javascript",   
    ".cjs": "javascript",  
    ".jsx": "javascript",   
    ".ts": "typescript",   
    ".tsx": "tsx",       
    ".java": "java",
    ".c": "cpp",    
    ".cc": "cpp",  
    ".cpp": "cpp", 
    ".cxx": "cpp", 
    ".h": "cpp",   
    ".hh": "cpp",  
    ".hpp": "cpp",  
}

PARSER_MAP = {
    "python": get_parser("python"),
    "javascript": get_parser("javascript"),
    "typescript": get_parser("typescript"),
    "tsx": get_parser("tsx"),
    "java": get_parser("java"),
    "cpp": get_parser("cpp"),
}


# this holds all the info about functions and classes we find 
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


# just a helper function i found online to make unique IDs 
def sha1_hex(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

# figures out what programming language a file is based on its extension 
def detect_language(path: Path) -> Optional[str]:
    lang = EXT_TO_LANG.get(path.suffix.lower())
    return lang if lang in PARSER_MAP else None

# super simple function that just grabs a piece of text from the source code 
def slice_text(src: bytes, node) -> bytes:
    return src[node.start_byte: node.end_byte]

# makes a unique ID for each piece of code we find 
def make_id(file: str, kind: str, name: str, start_line: int, end_line: int, text: bytes) -> str:
    key = f"{file}|{kind}|{name}|{start_line}|{end_line}|".encode() + text
    return sha1_hex(key)

# takes a list of strings and removes duplicates and sorts them
def uniq_sorted(xs: Iterable[str]) -> List[str]:
    return sorted(set(xs))


# this function finds python comments (the # ones) 
def leading_hash_comments_python(src: bytes, node) -> str:
    start_row = node.start_point[0]
    text = src.decode("utf-8", "ignore")
    lines = text.splitlines()
    i = start_row - 1
    buf = []
    
    while i >= 0:
        line = lines[i]
        if not line.strip():
            if buf: 
                break
            i -= 1
            continue
        if re.match(r"^\s*#", line):
            buf.append(re.sub(r"^\s*#\s?", "", line))
            i -= 1
            continue
        break  
        
    return "\n".join(reversed(buf)).strip()


# finds fancy comments in js/java/c++ (like /** */ and //)
def leading_docblock_or_slashes(src: bytes, node) -> str:
    pre = src[: node.start_byte].decode("utf-8", "ignore")
    
    m = re.search(r"/\*\*[\s\S]*?\*/\s*$", pre)
    if m:
        block = m.group(0)
        inner = re.sub(r"^/\*\*|\*/$", "", block, flags=re.DOTALL).strip()
        inner = re.sub(r"^[ \t]*\* ?", "", inner, flags=re.MULTILINE)
        return inner.strip()
        
    m2 = re.search(r"(?:^[ \t]*/{2,3}.*\n?)+\s*$", pre, flags=re.MULTILINE)
    if m2:
        lines = [re.sub(r"^[ \t]*/{2,3} ?", "", ln) 
                for ln in m2.group(0).splitlines() 
                if ln.strip()]
        return "\n".join(lines).strip()
        
    return ""

# main function for python files 
def python_extract_symbols(src: bytes, tree) -> List[SymbolRec]:
    root = tree.root_node
    recs: List[SymbolRec] = []

    # helper that converts the weird tree-sitter node text into normal strings
    # note: pls keep these types of functions in (there are more down below) to avoid unicode errors
    def node_text(n) -> str:
        return slice_text(src, n).decode("utf-8", "ignore")

    # finds those triple-quoted docstrings in python
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

    # gets the stuff inside the parentheses of functions 
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
            
            name_node = next((ch for ch in n.children if ch.type == "identifier"), None)
            name = node_text(name_node) if name_node else "<anonymous>"
            
            if kind == "function":
                pt = parms_text(n)
                signature = f"{name}{pt}" if pt else name
            else:
                bases = ""
                for ch in n.children:
                    if ch.type == "argument_list": 
                        bases = node_text(ch)
                        break
                signature = f"class {name}{bases}" if bases else f"class {name}"
            
            py_doc = first_docstring(n)
            lead = leading_hash_comments_python(src, n)
            doc_combined = (py_doc + ("\n" if py_doc and lead else "") + lead).strip()
            
            ids = collect_identifiers(n)  
            start_line = n.start_point[0] + 1
            end_line = n.end_point[0] + 1
            text = slice_text(src, n)  
            
            sid = make_id("<FILE>", kind, name, start_line, end_line, text)
            
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
                text=text.decode("utf-8", "ignore"),
            ))
            
        stack.extend(n.children)
        
    return recs

JS_FUNC_TYPES = {
    "function_declaration": "function",  
    "method_definition": "method",       
    "class_declaration": "class",        
}

TS_EXTRA_FUNC_TYPES = {
    "function_signature": "function",   
}

IDENTIFIER_TYPES_JS = {
    "identifier",                   
    "property_identifier",          
    "shorthand_property_identifier" 
}


# does the same thing as python_extract_symbols but for javascript/typescript 
def js_like_extract_symbols(src: bytes, tree) -> List[SymbolRec]:
    root = tree.root_node
    recs: List[SymbolRec] = []

    def node_text(n) -> str:
        return slice_text(src, n).decode("utf-8", "ignore")

    # grabs all the variable names and stuff
    def collect_identifiers(n) -> List[str]:
        ids = []
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur.type in IDENTIFIER_TYPES_JS:
                ids.append(node_text(cur).lower())
            stack.extend(cur.children)
        return uniq_sorted(ids)

    # gets the name of a function or class 
    # this can get super tricky bc of arrow functions and anonymous classes
    def name_for(n) -> str:
        for ch in n.children:
            if ch.type in IDENTIFIER_TYPES_JS:
                return node_text(ch)
        return next((node_text(ch) for ch in n.children 
                    if ch.type in IDENTIFIER_TYPES_JS), "<anonymous>")

    # extracts parameters from functions
    def params_text(n) -> str:
        for ch in n.children:
            if ch.type in {"formal_parameters",
                          "arguments",
                          "type_parameters"}:
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
            
        if kind:
            name = name_for(n)
            pt = params_text(n)
            signature = f"{name}{pt}" if pt else name
            
            ids = collect_identifiers(n)
            start_line = n.start_point[0] + 1
            end_line = n.end_point[0] + 1
            text = slice_text(src, n)
            
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
                text=text.decode("utf-8", "ignore"),
            ))
            
        stack.extend(n.children)
        
    return recs

IDENTIFIER_TYPES_JAVA = {
    "identifier",       
    "type_identifier"   
}


# java parser
# honestly java is the most straightforward one because everything has to be in a class
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

    # gets names of java stuff, way simpler since java is really organized
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
            name = name_for(n)
            
            pt = params_text(n)
            signature = f"{name}{pt}" if pt else name
            
            ids = collect_identifiers(n)
            start_line = n.start_point[0] + 1
            end_line = n.end_point[0] + 1
            text = slice_text(src, n)
            
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
                text=text.decode("utf-8", "ignore"),
            ))
            
        stack.extend(n.children)
        
    return recs

IDENTIFIER_TYPES_CPP = {
    "identifier",       
    "type_identifier",   
    "field_identifier"  
}


# c++ parser
# hardest one bc c++ has so many weird features
def cpp_extract_symbols(src: bytes, tree) -> List[SymbolRec]:
    root = tree.root_node
    recs: List[SymbolRec] = []

    # more text decoding that handles most cases
    def node_text(n) -> str:
        return slice_text(src, n).decode("utf-8", "ignore")

    # identifier collector for c++ 
    # had to handle normal variables, types, and pesky template parameters
    def collect_identifiers(n) -> List[str]:
        ids = []
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur.type in IDENTIFIER_TYPES_CPP:
                ids.append(node_text(cur).lower())
            stack.extend(cur.children)
        return uniq_sorted(ids)

    # checks if something looks like a function 
    def function_like(n) -> bool:
        if n.type == "function_definition":
            return True
        if n.type == "declaration":
            return any(ch.type == "function_declarator" for ch in n.children)
        return False

    # c++ functions can be super weird with templates and namespaces 
    def name_and_sig_for_func(n) -> tuple[str, str]:
        # helper that digs through the mess of c++ syntax to find the actual function name
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

    TARGETS = {
        "class_specifier": "class",     
        "struct_specifier": "struct",    
    }

    stack = [root]
    while stack:
        n = stack.pop()
        kind = None
        name = ""
        signature = ""
        
        if n.type in TARGETS:
            kind = TARGETS[n.type]
            name_node = next((ch for ch in n.children if ch.type == "type_identifier"), None)
            name = node_text(name_node) if name_node else "<anonymous>"
            signature = f"{kind} {name}"
            
        elif function_like(n):
            kind = "function"
            name, signature = name_and_sig_for_func(n)
            
        if kind:
            ids = collect_identifiers(n) 
            start_line = n.start_point[0] + 1  
            end_line = n.end_point[0] + 1
            text = slice_text(src, n) 
            
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
                text=text.decode("utf-8", "ignore"),
            ))
            
        stack.extend(n.children)
        
    return recs



# This is like the manager function, figures out what kind of file it is and calls the right parser
def parse_file(path: Path) -> List[SymbolRec]:
    lang_key = detect_language(path)
    if not lang_key:
        return []  
    parser = PARSER_MAP[lang_key]

    try:
        src = path.read_bytes()
    except Exception:
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
        r.file = str(path.as_posix())  
        r.id = make_id(r.file, r.kind, r.name, r.start_line, r.end_line, r.text.encode("utf-8"))
    return recs


# big boss function that walks through all the files in a project
# note: it skips stuff like node_modules bc those are just dependencies
def crawl_repo(root: Path) -> List[SymbolRec]:
    symbols: List[SymbolRec] = []
    
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {
            ".git",
            "node_modules",
            "dist", "build", "out",
            "cmake-build-debug",
            "cmake-build-release",
            "__pycache__",
            ".venv", "venv",
            ".mypy_cache",
            ".pytest_cache"
        }]
        
        for fn in filenames:
            p = Path(dirpath) / fn
            if detect_language(p):
                symbols.extend(parse_file(p))
                
    return symbols


# main function that ties it all together 
def main():
    ap = argparse.ArgumentParser(description="EP2C – Step 2 (Tree‑sitter) repo symbol extractor")
    ap.add_argument("root", type=str, help="Path to repository root")
    ap.add_argument("--out", type=str, default="example_symbols.json", help="Output JSON path")
    ap.add_argument("--emit-module-span", action="store_true", help="Also emit a file‑level 'module' span per file")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"error: {root} is not a directory")

    symbols = crawl_repo(root)

    if args.emit_module_span:
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
            m = re.match(r"^(?:\s*(?:#|//).*\n|/\*[\s\S]*?\*/\s*\n)+", src)
            if m:
                block = m.group(0)
                block = re.sub(r"^\s*#\s?", "", block, flags=re.MULTILINE)     
                block = re.sub(r"^\s*//\s?", "", block, flags=re.MULTILINE)   
                block = re.sub(r"/\*|\*/", "", block)                          
                header = block.strip()
                
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
                text=src,
            ))

    out = [asdict(s) for s in symbols]
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out} with {len(symbols)} symbols")

if __name__ == "__main__":
    main()