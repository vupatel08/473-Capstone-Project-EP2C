#!/usr/bin/env python3
"""
EP2C – Step 2 (Tree‑sitter): Parse a code repository into symbol records

What this script does (as per README spec):
  • Walk a given repo directory.
  • Detect language from file extension.
  • Parse each file with Tree‑sitter to extract symbols (functions, classes, methods, and top‑level code spans where sensible).
  • For each symbol, build a record with fields:
      {id, file, kind, name, signature, docstring, identifiers, start_line, end_line, text}
  • identifiers is a bag‑of‑words (unique identifiers used inside the symbol).

Supported languages: Python, JavaScript, TypeScript/TSX, Java, C/C++ (headers too).

Usage:
    pip install "tree_sitter==0.21.3" "tree_sitter_languages>=1.10,<2"
    python repo_symbols_treesitter.py /path/to/repo --out symbols.json

Notes:
  • Uses the new `get_parser()` API from tree_sitter_languages.
  • Doc/comments capture:
      - Python: real docstring + leading `#` run above the def/class.
      - JS/TS/TSX: JSDoc `/** ... */` or trailing `//` run immediately above.
      - Java: Javadoc `/** ... */` or `//` run above methods/ctors/classes.
      - C/C++: Doxygen/Javadoc style `/** ... */` or consecutive `//`/`///` lines above functions/classes.
  • Signature is heuristic: parameter lists where available; otherwise the name or sliced declarator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tree_sitter_languages import get_parser

# ----------------------------
# Language Detection Constants
# ----------------------------

# Maps file extensions to their corresponding language identifiers
# This mapping is used to determine which parser to use for each file
# Note: C files are treated as C++ for symbol extraction since the grammar is similar
EXT_TO_LANG = {
    # Python files
    ".py": "python",
    
    # JavaScript family
    ".js": "javascript",    # Standard JavaScript
    ".mjs": "javascript",   # ES6 modules
    ".cjs": "javascript",   # CommonJS modules
    ".jsx": "javascript",   # React JavaScript
    ".ts": "typescript",    # TypeScript
    ".tsx": "tsx",         # TypeScript with React
    
    # Java files
    ".java": "java",
    
    # C/C++ family
    ".c": "cpp",    # C source files (parsed as C++)
    ".cc": "cpp",   # C++ source files
    ".cpp": "cpp",  # C++ source files
    ".cxx": "cpp",  # C++ source files
    ".h": "cpp",    # C/C++ header files
    ".hh": "cpp",   # C++ header files
    ".hpp": "cpp",  # C++ header files
}

# Parsers: get_parser returns a configured Parser for that language
PARSER_MAP = {
    "python": get_parser("python"),
    "javascript": get_parser("javascript"),
    "typescript": get_parser("typescript"),
    "tsx": get_parser("tsx"),
    "java": get_parser("java"),
    "cpp": get_parser("cpp"),
}

# ----------------------------
# Symbol Record Definition
# ----------------------------
@dataclass
class SymbolRec:
    """
    Represents a code symbol (function, class, method, etc.) extracted from source code.
    
    This class stores all relevant information about a code symbol, including its
    location, content, and metadata. It's designed to be easily serializable to JSON.
    
    Attributes:
        id (str): Unique identifier (SHA1 hash of content)
        file (str): Path to the source file containing this symbol
        kind (str): Type of symbol (function, class, method, etc.)
        name (str): Symbol name as it appears in code
        signature (str): Full signature including parameters/return type
        docstring (str): Documentation comments associated with the symbol
        identifiers (List[str]): List of identifiers used within the symbol
        start_line (int): First line number in source file (1-based)
        end_line (int): Last line number in source file (1-based)
        text (str): Full source code text of the symbol
    """
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

# ----------------------------
# Utility Helper Functions
# ----------------------------

def sha1_hex(b: bytes) -> str:
    """
    Generate a SHA1 hex digest of the input bytes.
    
    Args:
        b (bytes): Input data to hash
        
    Returns:
        str: Hexadecimal string of the SHA1 hash
    """
    return hashlib.sha1(b).hexdigest()

def detect_language(path: Path) -> Optional[str]:
    """
    Determine the programming language of a file based on its extension.
    
    Args:
        path (Path): Path to the source code file
        
    Returns:
        Optional[str]: Language identifier if supported, None otherwise
        
    Example:
        >>> detect_language(Path("example.py"))
        'python'
        >>> detect_language(Path("example.txt"))
        None
    """
    lang = EXT_TO_LANG.get(path.suffix.lower())
    return lang if lang in PARSER_MAP else None

def slice_text(src: bytes, node) -> bytes:
    """
    Extract a portion of source code corresponding to a syntax tree node.
    
    Args:
        src (bytes): Complete source code as bytes
        node: Tree-sitter AST node
        
    Returns:
        bytes: Source code fragment for the node
    """
    return src[node.start_byte: node.end_byte]

def make_id(file: str, kind: str, name: str, start_line: int, end_line: int, text: bytes) -> str:
    """
    Generate a unique identifier for a code symbol.
    
    Creates a deterministic ID by combining multiple attributes of the symbol
    and hashing them together. This ensures the same symbol will always get
    the same ID, even across different runs.
    
    Args:
        file (str): Source file path
        kind (str): Symbol type (function, class, etc.)
        name (str): Symbol name
        start_line (int): Starting line number
        end_line (int): Ending line number
        text (bytes): Symbol's source code
        
    Returns:
        str: Unique identifier (SHA1 hash)
    """
    key = f"{file}|{kind}|{name}|{start_line}|{end_line}|".encode() + text
    return sha1_hex(key)

def uniq_sorted(xs: Iterable[str]) -> List[str]:
    """
    Create a sorted list of unique items from an iterable.
    
    Args:
        xs (Iterable[str]): Input sequence of strings
        
    Returns:
        List[str]: Sorted list with duplicates removed
        
    Example:
        >>> uniq_sorted(['b', 'a', 'b', 'c'])
        ['a', 'b', 'c']
    """
    return sorted(set(xs))


def leading_hash_comments_python(src: bytes, node) -> str:
    """
    Extract Python comment blocks (#) that appear immediately before a node.
    
    This function looks for consecutive comment lines that appear directly
    above a Python function or class definition. It preserves multi-line
    comments while removing the '#' markers and leading whitespace.
    
    Args:
        src (bytes): Complete source code of the file
        node: Tree-sitter AST node (typically a function or class definition)
        
    Returns:
        str: Combined comment text without '#' markers
        
    Example:
        For source code like:
            # This is a comment
            # that spans multiple lines
            def some_function():
                pass
                
        Returns: "This is a comment\\nthat spans multiple lines"
    """
    start_row = node.start_point[0]
    text = src.decode("utf-8", "ignore")
    lines = text.splitlines()
    i = start_row - 1
    buf = []
    
    while i >= 0:
        line = lines[i]
        # Handle blank lines
        if not line.strip():
            if buf:  # Stop if we've found comments and hit a blank line
                break
            i -= 1
            continue
        # Collect comment lines
        if re.match(r"^\s*#", line):
            # Remove the # and one optional space after it
            buf.append(re.sub(r"^\s*#\s?", "", line))
            i -= 1
            continue
        break  # Stop at first non-comment line
        
    return "\n".join(reversed(buf)).strip()


def leading_docblock_or_slashes(src: bytes, node) -> str:
    """
    Extract documentation blocks in C-style languages (Java, JavaScript, C++).
    
    Supports two main documentation styles:
    1. Block comments: /** ... */  (Javadoc/JSDoc/Doxygen style)
    2. Line comments: // or /// (consecutive lines only)
    
    Args:
        src (bytes): Complete source code of the file
        node: Tree-sitter AST node (typically a function or class definition)
        
    Returns:
        str: Extracted documentation with comment markers removed
        
    Examples:
        For JSDoc/Javadoc style:
            /**
             * This is a documentation block
             * @param {string} name - Description
             */
            function example() {}
            
        For line comments:
            // This is a documentation block
            // that uses line comments
            function example() {}
            
    Note:
        - For block comments, removes the /** and */ markers and * line prefixes
        - For line comments, combines consecutive comment lines
        - Preserves internal formatting and whitespace
    """
    pre = src[: node.start_byte].decode("utf-8", "ignore")
    
    # Try to find a /** ... */ block comment first
    m = re.search(r"/\*\*[\s\S]*?\*/\s*$", pre)
    if m:
        block = m.group(0)
        # Remove the opening /** and closing */
        inner = re.sub(r"^/\*\*|\*/$", "", block, flags=re.DOTALL).strip()
        # Remove * prefix from each line while preserving indentation
        inner = re.sub(r"^[ \t]*\* ?", "", inner, flags=re.MULTILINE)
        return inner.strip()
        
    # Fallback: look for consecutive //, /// comments
    m2 = re.search(r"(?:^[ \t]*/{2,3}.*\n?)+\s*$", pre, flags=re.MULTILINE)
    if m2:
        # Remove comment markers while preserving the comment text
        lines = [re.sub(r"^[ \t]*/{2,3} ?", "", ln) 
                for ln in m2.group(0).splitlines() 
                if ln.strip()]
        return "\n".join(lines).strip()
        
    return ""

# ----------------------------
# Per‑language symbol extraction
# ----------------------------
# We use language‑specific node kinds from Tree‑sitter grammars.

# ---------- Python Parser ----------

def python_extract_symbols(src: bytes, tree) -> List[SymbolRec]:
    """
    Extract functions and classes from Python source code using Tree-sitter AST.
    
    This function walks the Python syntax tree to find:
    - Function definitions (including methods)
    - Class definitions
    - Associated documentation (docstrings and comments)
    
    Args:
        src (bytes): Source code of the Python file
        tree: Tree-sitter AST root
        
    Returns:
        List[SymbolRec]: List of extracted symbols with metadata
    """
    root = tree.root_node
    recs: List[SymbolRec] = []

    def node_text(n) -> str:
        """Extract text content from an AST node."""
        return slice_text(src, n).decode("utf-8", "ignore")

    def first_docstring(n) -> str:
        """
        Extract the first docstring from a function/class body.
        
        Looks for a string literal as the first expression in a 
        function/class body, which is Python's standard docstring pattern.
        
        Example:
            def func():
                '''This is a docstring'''
                pass
        """
        for child in n.children:
            # Look in the function/class body block
            if child.type in {"block", "suite"} and child.children:
                first = child.children[0]
                # Check if first statement is a string literal
                if first.type == "expression_statement" and first.children and \
                   first.children[0].type in {"string", "string_literal"}:
                    return node_text(first.children[0]).strip("'\" ")
        return ""

    def collect_identifiers(n) -> List[str]:
        """
        Collect all identifiers used within a node's scope.
        
        Walks the AST to find all variable names, function names,
        class names, etc. that are used within this symbol's definition.
        These are used for symbol cross-referencing and analysis.
        """
        ids = []
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur.type == "identifier":
                ids.append(node_text(cur).lower())
            stack.extend(cur.children)
        return uniq_sorted(ids)

    def parms_text(n) -> str:
        """
        Extract the parameter list from a function definition.
        
        Example:
            def func(a: int, b: str = "default") -> None:
            Returns: "(a: int, b: str = "default")"
        """
        for ch in n.children:
            if ch.type in {"parameters", "lambda_parameters"}:
                return node_text(ch)
        return ""

    # Walk the AST looking for function and class definitions
    stack = [root]
    while stack:
        n = stack.pop()
        
        # Process function and class definitions
        if n.type in {"function_definition", "class_definition"}:
            # Determine symbol type
            kind = "function" if n.type == "function_definition" else "class"
            
            # Extract the symbol's name
            name_node = next((ch for ch in n.children if ch.type == "identifier"), None)
            name = node_text(name_node) if name_node else "<anonymous>"
            
            # Build the signature based on symbol type
            if kind == "function":
                # For functions, include parameter list
                pt = parms_text(n)
                signature = f"{name}{pt}" if pt else name
            else:
                # For classes, include base classes if any
                bases = ""
                for ch in n.children:
                    if ch.type == "argument_list":  # Base class list
                        bases = node_text(ch)
                        break
                signature = f"class {name}{bases}" if bases else f"class {name}"
            
            # Combine docstring and leading comments
            py_doc = first_docstring(n)
            lead = leading_hash_comments_python(src, n)
            doc_combined = (py_doc + ("\n" if py_doc and lead else "") + lead).strip()
            
            # Collect metadata
            ids = collect_identifiers(n)  # Used identifiers within the symbol
            start_line = n.start_point[0] + 1  # Convert to 1-based line numbers
            end_line = n.end_point[0] + 1
            text = slice_text(src, n)  # Full source text
            
            # Generate unique ID for this symbol
            sid = make_id("<FILE>", kind, name, start_line, end_line, text)
            
            # Create and store the symbol record
            recs.append(SymbolRec(
                id=sid,
                file="",  # File path is filled in later
                kind=kind,
                name=name,
                signature=signature,
                docstring=doc_combined,
                identifiers=ids,
                start_line=start_line,
                end_line=end_line,
                text=text.decode("utf-8", "ignore"),
            ))
            
        # Continue walking the tree
        stack.extend(n.children)
        
    return recs

# ---------- JavaScript / TypeScript / TSX Parser ----------

# Map of Tree-sitter node types to our symbol kinds for JavaScript
JS_FUNC_TYPES = {
    "function_declaration": "function",  # function foo() {}
    "method_definition": "method",       # class X { foo() {} }
    "class_declaration": "class",        # class X {}
}

# Additional node types specific to TypeScript
TS_EXTRA_FUNC_TYPES = {
    "function_signature": "function",    # interface X { foo(): void }
}

# All possible identifier node types in JS/TS
IDENTIFIER_TYPES_JS = {
    "identifier",                    # regular identifiers
    "property_identifier",           # object property names
    "shorthand_property_identifier"  # { foo } shorthand properties
}


def js_like_extract_symbols(src: bytes, tree) -> List[SymbolRec]:
    """
    Extract symbols from JavaScript, TypeScript, and TSX code.
    
    Handles the complexities of modern JS ecosystem:
    - ES6+ syntax (classes, methods, arrow functions)
    - TypeScript type annotations and interfaces
    - JSX/TSX mixed markup syntax
    
    Args:
        src (bytes): Source code of the JS/TS file
        tree: Tree-sitter AST root
        
    Returns:
        List[SymbolRec]: List of extracted symbols with metadata
    """
    root = tree.root_node
    recs: List[SymbolRec] = []

    def node_text(n) -> str:
        """Extract text from an AST node, handling UTF-8 encoding."""
        return slice_text(src, n).decode("utf-8", "ignore")

    def collect_identifiers(n) -> List[str]:
        """
        Collect all identifiers used within a symbol's scope.
        
        Handles JS/TS specific identifier types:
        - Regular variable/function names
        - Object property names
        - Shorthand object properties
        
        Example:
            function foo(x) {
                return { bar, baz: 42 };
            }
            Collects: ['foo', 'x', 'bar', 'baz']
        """
        ids = []
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur.type in IDENTIFIER_TYPES_JS:
                ids.append(node_text(cur).lower())
            stack.extend(cur.children)
        return uniq_sorted(ids)

    def name_for(n) -> str:
        """
        Extract the name of a symbol from its AST node.
        
        Handles various ways names can appear in JS/TS:
        - Direct identifiers: function foo()
        - Property names: class { foo() {} }
        - Computed names: class { [foo]() {} }
        
        Returns '<anonymous>' for unnamed functions/classes.
        """
        for ch in n.children:
            if ch.type in IDENTIFIER_TYPES_JS:
                return node_text(ch)
        return next((node_text(ch) for ch in n.children 
                    if ch.type in IDENTIFIER_TYPES_JS), "<anonymous>")

    def params_text(n) -> str:
        """
        Extract parameter list or type parameters from a symbol.
        
        Handles:
        - Function parameters: (a, b)
        - Type parameters: <T>
        - Constructor arguments: new Class(x, y)
        """
        for ch in n.children:
            if ch.type in {"formal_parameters",    # Regular params
                          "arguments",             # Constructor args
                          "type_parameters"}:      # Generic type params
                return node_text(ch)
        return ""

    # Walk the AST looking for functions, methods, and classes
    stack = [root]
    while stack:
        n = stack.pop()
        
        # Determine the kind of symbol we're looking at
        kind = None
        if n.type in JS_FUNC_TYPES:
            # Standard JavaScript constructs
            kind = JS_FUNC_TYPES[n.type]
        elif n.type in TS_EXTRA_FUNC_TYPES:
            # TypeScript-specific constructs
            kind = TS_EXTRA_FUNC_TYPES[n.type]
            
        if kind:
            # Extract basic symbol information
            name = name_for(n)
            pt = params_text(n)
            signature = f"{name}{pt}" if pt else name
            
            # Collect metadata
            ids = collect_identifiers(n)
            start_line = n.start_point[0] + 1
            end_line = n.end_point[0] + 1
            text = slice_text(src, n)
            
            # Extract JSDoc or regular comments
            doc_or_jsdoc = leading_docblock_or_slashes(src, n)
            
            # Create symbol record
            recs.append(SymbolRec(
                id="",  # Will be set later
                file="",  # Will be set later
                kind=kind,
                name=name,
                signature=signature,
                docstring=doc_or_jsdoc,
                identifiers=ids,
                start_line=start_line,
                end_line=end_line,
                text=text.decode("utf-8", "ignore"),
            ))
            
        # Continue walking the tree
        stack.extend(n.children)
        
    return recs

# ---------- Java Parser ----------

# Types of nodes that can be identifiers in Java
IDENTIFIER_TYPES_JAVA = {
    "identifier",       # Variable and method names
    "type_identifier"   # Class and interface names
}


def java_extract_symbols(src: bytes, tree) -> List[SymbolRec]:
    """
    Extract symbols from Java source code.
    
    Handles Java-specific constructs:
    - Classes and interfaces
    - Methods and constructors
    - Field declarations
    - Generic type parameters
    - Annotations
    
    Special handling for:
    - Javadoc comments (/** ... */)
    - Package and import statements
    - Access modifiers and other metadata
    
    Args:
        src (bytes): Source code of the Java file
        tree: Tree-sitter AST root
        
    Returns:
        List[SymbolRec]: List of extracted symbols with metadata
    """
    root = tree.root_node
    recs: List[SymbolRec] = []

    def node_text(n) -> str:
        """Extract text from an AST node, handling UTF-8 encoding."""
        return slice_text(src, n).decode("utf-8", "ignore")

    def collect_identifiers(n) -> List[str]:
        """
        Collect all identifiers used within a Java symbol's scope.
        
        Captures both regular identifiers and type identifiers:
        - Method and variable names
        - Class and interface names
        - Generic type parameters
        
        Example:
            class MyClass<T> extends BaseClass {
                private List<String> items;
            }
            Collects: ['myclass', 't', 'baseclass', 'list', 'string', 'items']
        """
        ids = []
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur.type in IDENTIFIER_TYPES_JAVA:
                ids.append(node_text(cur).lower())
            stack.extend(cur.children)
        return uniq_sorted(ids)

    def name_for(n) -> str:
        """
        Extract the name of a Java symbol from its AST node.
        
        Handles:
        - Class and interface names
        - Method names
        - Constructor names
        - Anonymous inner classes
        
        Returns '<anonymous>' for unnamed classes.
        """
        for ch in n.children:
            if ch.type in IDENTIFIER_TYPES_JAVA:
                return node_text(ch)
        return "<anonymous>"

    def params_text(n) -> str:
        """
        Extract parameter list from a method or constructor.
        
        Handles:
        - Method parameters with types
        - Generic type parameters
        - Varargs parameters
        - Final parameters
        
        Example:
            void method(final String name, int... numbers)
        """
        for ch in n.children:
            if ch.type in {"formal_parameters"}:
                return node_text(ch)
        return ""

    # Map of Java AST node types to our symbol kinds
    TARGETS = {
        "class_declaration": "class",            # Regular classes
        "interface_declaration": "interface",    # Interfaces
        "enum_declaration": "enum",             # Enum types
        "method_declaration": "method",         # Methods
        "constructor_declaration": "constructor", # Constructors
    }

    # Walk the AST looking for classes, interfaces, methods, etc.
    stack = [root]
    while stack:
        n = stack.pop()
        
        # Process known Java constructs
        if n.type in TARGETS:
            # Get symbol type and basic info
            kind = TARGETS[n.type]
            name = name_for(n)
            
            # Build appropriate signature based on symbol type
            pt = params_text(n)
            signature = f"{name}{pt}" if pt else name
            
            # Collect metadata
            ids = collect_identifiers(n)
            start_line = n.start_point[0] + 1
            end_line = n.end_point[0] + 1
            text = slice_text(src, n)
            
            # Extract Javadoc or regular comments
            jdoc = leading_docblock_or_slashes(src, n)
            
            # Create symbol record
            recs.append(SymbolRec(
                id="",  # Will be set later
                file="",  # Will be set later
                kind=kind,
                name=name,
                signature=signature,
                docstring=jdoc,
                identifiers=ids,
                start_line=start_line,
                end_line=end_line,
                text=text.decode("utf-8", "ignore"),
            ))
            
        # Continue walking the tree
        stack.extend(n.children)
        
    return recs

# ---------- C/C++ Parser ----------

# Types of nodes that can be identifiers in C++
IDENTIFIER_TYPES_CPP = {
    "identifier",        # Variable and function names
    "type_identifier",   # Class and type names
    "field_identifier"   # Struct/class field names
}


def cpp_extract_symbols(src: bytes, tree) -> List[SymbolRec]:
    """
    Extract symbols from C/C++ source code.
    
    Handles the complexities of C++ syntax:
    - Functions and member functions
    - Classes and structs
    - Templates and specializations
    - Operator overloading
    - Namespace constructs
    
    Special handling for:
    - Header files (.h, .hpp)
    - Template declarations
    - Friend declarations
    - Nested classes
    - Multiple inheritance
    - Doxygen comments
    
    Args:
        src (bytes): Source code of the C/C++ file
        tree: Tree-sitter AST root
        
    Returns:
        List[SymbolRec]: List of extracted symbols with metadata
        
    Note:
        C code is parsed using the C++ parser since the grammars
        are similar enough for symbol extraction purposes.
    """
    root = tree.root_node
    recs: List[SymbolRec] = []

    def node_text(n) -> str:
        """Extract text from an AST node, handling UTF-8 encoding."""
        return slice_text(src, n).decode("utf-8", "ignore")

    def collect_identifiers(n) -> List[str]:
        """
        Collect all identifiers used within a C++ symbol's scope.
        
        Captures various C++ identifier types:
        - Variable and function names
        - Class and type names
        - Template parameters
        - Namespace identifiers
        - Member variable names
        
        Example:
            template<typename T>
            class MyClass : public BaseClass {
                std::vector<int> items;
            };
            Collects: ['t', 'myclass', 'baseclass', 'std', 'vector', 'int', 'items']
        """
        ids = []
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur.type in IDENTIFIER_TYPES_CPP:
                ids.append(node_text(cur).lower())
            stack.extend(cur.children)
        return uniq_sorted(ids)

    def function_like(n) -> bool:
        """
        Determine if a node represents a function-like declaration.
        
        Handles:
        - Regular function definitions
        - Member function declarations
        - Operator overloads
        - Template functions
        - Function declarations in headers
        """
        if n.type == "function_definition":
            return True
        if n.type == "declaration":
            return any(ch.type == "function_declarator" for ch in n.children)
        return False

    def name_and_sig_for_func(n) -> tuple[str, str]:
        """
        Extract both name and full signature from a function node.
        
        Handles complex C++ function declarations:
        - Template functions
        - Member functions
        - Operator overloads
        - Function pointers
        - Nested functions
        
        Returns:
            tuple[str, str]: (function_name, full_signature)
            
        Example:
            Input: "template<class T> void MyClass<T>::func(int x) const"
            Returns: ("func", "void MyClass<T>::func(int x) const")
        """
        def find_ident(m):
            """Find the main identifier in a function declarator."""
            stack = [m]
            while stack:
                cur = stack.pop()
                if cur.type in {"identifier"}:
                    return node_text(cur)
                stack.extend(cur.children)
            return "<anonymous>"
            
        # Handle function definitions
        if n.type == "function_definition":
            decl = next((ch for ch in n.children if ch.type.endswith("declarator")), None)
            if decl:
                name = find_ident(decl)
                sig = node_text(decl)
                return name, sig
                
        # Handle function declarations
        decl = next((ch for ch in n.children if ch.type == "function_declarator"), None)
        if decl:
            name = find_ident(decl)
            sig = node_text(decl)
            return name, sig
            
        return "<anonymous>", node_text(n)

    # Map of C++ AST node types to our symbol kinds
    TARGETS = {
        "class_specifier": "class",      # class declarations
        "struct_specifier": "struct",     # struct declarations
    }

    # Main C++ AST traversal loop
    stack = [root]
    while stack:
        n = stack.pop()
        kind = None
        name = ""
        signature = ""
        
        # Handle class and struct declarations
        if n.type in TARGETS:
            kind = TARGETS[n.type]
            # Extract type name from the class/struct declaration
            name_node = next((ch for ch in n.children if ch.type == "type_identifier"), None)
            name = node_text(name_node) if name_node else "<anonymous>"
            signature = f"{kind} {name}"
            
        # Handle function declarations and definitions
        elif function_like(n):
            kind = "function"
            name, signature = name_and_sig_for_func(n)
            
        # Create symbol record if we found a valid declaration
        if kind:
            # Collect metadata
            ids = collect_identifiers(n)  # Used identifiers within symbol
            start_line = n.start_point[0] + 1  # Convert to 1-based line numbers
            end_line = n.end_point[0] + 1
            text = slice_text(src, n)  # Full source text
            
            # Extract documentation comments (Doxygen style)
            cdoc = leading_docblock_or_slashes(src, n)
            
            # Create and store the symbol record
            recs.append(SymbolRec(
                id="",  # Will be set later with file info
                file="",  # Will be set later
                kind=kind,
                name=name,
                signature=signature,
                docstring=cdoc,
                identifiers=ids,
                start_line=start_line,
                end_line=end_line,
                text=text.decode("utf-8", "ignore"),
            ))
            
        # Continue traversing the AST
        stack.extend(n.children)
        
    return recs

# ----------------------------
# Repo crawl & orchestration
# ----------------------------

def parse_file(path: Path) -> List[SymbolRec]:
    """
    Parse a single source code file and extract all symbols.
    
    This is the main orchestration function that:
    1. Detects the file's language based on extension
    2. Reads and parses the file using Tree-sitter
    3. Delegates to language-specific extractors
    4. Post-processes records with file info and IDs
    
    Supports multiple languages:
    - Python (.py)
    - JavaScript/TypeScript (.js, .ts, .tsx)
    - Java (.java)
    - C/C++ (.c, .cpp, .h, etc.)
    
    Args:
        path (Path): Path to the source code file
        
    Returns:
        List[SymbolRec]: List of extracted symbol records
        
    Note:
        - Skips unsupported file types
        - Handles read errors gracefully
        - File paths are normalized to POSIX format
        - Generated IDs are deterministic based on content
    """
    # Detect language and get appropriate parser
    lang_key = detect_language(path)
    if not lang_key:
        return []  # Unsupported file type
    parser = PARSER_MAP[lang_key]

    # Read file contents safely
    try:
        src = path.read_bytes()
    except Exception:
        return []  # Skip files we can't read

    # Parse file into AST
    tree = parser.parse(src)

    # Delegate to language-specific extractors
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

    # Post-process records with file info and IDs
    for r in recs:
        r.file = str(path.as_posix())  # Normalize path format
        r.id = make_id(r.file, r.kind, r.name, r.start_line, r.end_line, r.text.encode("utf-8"))
    return recs


def crawl_repo(root: Path) -> List[SymbolRec]:
    """
    Recursively crawl a code repository and extract symbols from all supported files.
    
    This function:
    1. Walks the repository directory tree
    2. Filters out common build/cache directories
    3. Processes all supported source files
    4. Aggregates symbols from all files
    
    Excluded directories:
    - Version control: .git
    - Package managers: node_modules
    - Build output: dist, build, out
    - CMake builds: cmake-build-*
    - Python cache: __pycache__, .pytest_cache
    - Virtual envs: .venv, venv
    - Type checking: .mypy_cache
    
    Args:
        root (Path): Path to the root of the repository
        
    Returns:
        List[SymbolRec]: Combined list of all symbols found
        
    Note:
        The function is aware of common patterns in modern development
        workflows and skips directories that typically contain generated
        code, dependencies, or cache files.
    """
    symbols: List[SymbolRec] = []
    
    # Walk directory tree
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out build/cache directories in-place
        # This modifies the dirnames list to prevent recursing into these dirs
        dirnames[:] = [d for d in dirnames if d not in {
            # Version control
            ".git",
            # Package management
            "node_modules",
            # Build artifacts
            "dist", "build", "out",
            # CMake build directories
            "cmake-build-debug",
            "cmake-build-release",
            # Python specific
            "__pycache__",
            ".venv", "venv",
            ".mypy_cache",
            ".pytest_cache"
        }]
        
        # Process each file in the current directory
        for fn in filenames:
            p = Path(dirpath) / fn
            # Only process files with supported languages
            if detect_language(p):
                symbols.extend(parse_file(p))
                
    return symbols

# ----------------------------
# CLI Implementation
# ----------------------------

def main():
    """
    Command-line interface for the repository symbol extractor.
    
    Features:
    1. Process a repository and extract symbols from supported files
    2. Optional module-level symbol generation
    3. Output results in JSON format
    
    Command-line Arguments:
        root: Path to repository root directory
        --out: Output JSON file path (default: example_symbols.json)
        --emit-module-span: Generate file-level module symbols
        
    Example Usage:
        python repo_parser.py /path/to/repo --out symbols.json
        python repo_parser.py /path/to/repo --emit-module-span
        
    The output JSON format is an array of symbol records with fields:
        id: Unique identifier
        file: Source file path
        kind: Symbol type (function, class, etc.)
        name: Symbol name
        signature: Full signature or declaration
        docstring: Associated documentation
        identifiers: Used identifiers
        start_line: Starting line number
        end_line: Ending line number
        text: Full source text
    """
    # Set up command-line argument parsing
    ap = argparse.ArgumentParser(description="EP2C – Step 2 (Tree‑sitter) repo symbol extractor")
    ap.add_argument("root", type=str, help="Path to repository root")
    ap.add_argument("--out", type=str, default="example_symbols.json", help="Output JSON path")
    ap.add_argument("--emit-module-span", action="store_true", help="Also emit a file‑level 'module' span per file")
    args = ap.parse_args()

    # Validate repository path
    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"error: {root} is not a directory")

    # Extract symbols from the repository
    symbols = crawl_repo(root)

    # Optionally generate module-level symbols
    if args.emit_module_span:
        # Group symbols by file
        by_file: Dict[str, List[SymbolRec]] = {}
        for s in symbols:
            by_file.setdefault(s.file, []).append(s)
            
        # Process each file
        for file, _ in by_file.items():
            try:
                # Read file contents
                src = Path(file).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
                
            # Determine file span
            start = 1
            end = src.count("\n") + 1
            
            # Create unique ID for module
            sid = sha1_hex((file + "|module").encode())
            
            # Extract file-level documentation
            header = ""
            # Look for leading comments in any style
            m = re.match(r"^(?:\s*(?:#|//).*\n|/\*[\s\S]*?\*/\s*\n)+", src)
            if m:
                block = m.group(0)
                # Clean up comment markers while preserving content
                block = re.sub(r"^\s*#\s?", "", block, flags=re.MULTILINE)      # Python style
                block = re.sub(r"^\s*//\s?", "", block, flags=re.MULTILINE)     # C++ style
                block = re.sub(r"/\*|\*/", "", block)                           # Block style
                header = block.strip()
                
            # Create and add module-level symbol
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

    # Convert records to dictionaries and write JSON output
    out = [asdict(s) for s in symbols]
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out} with {len(symbols)} symbols")

if __name__ == "__main__":
    main()
