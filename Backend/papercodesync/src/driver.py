#!/usr/bin/env python3
"""
PCS Pipeline
-------------
Converts a parsed paper Markdown + code repository into synced artifacts:
  1) chunks.json
  2) symbols.json
  3) matches.jsonl
"""

import subprocess
import shutil
from pathlib import Path
import sys
import os

HERE = Path(__file__).resolve().parent
PY = sys.executable

# Internal script locations
CREATE_CHUNKS  = (HERE / "create_chunks.py").resolve()
CREATE_SYMBOLS = (HERE / "create_symbols.py").resolve()
CREATE_MAP     = (HERE / "create_map.py").resolve()

# Default output folder
DATA_DIR       = (HERE / "../data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_JSON  = DATA_DIR / "chunks.json"
SYMBOLS_JSON = DATA_DIR / "symbols.json"
MATCHES_JSON = DATA_DIR / "matches.jsonl"

def delete_files_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath):
            os.remove(filepath)
            print(f"Deleted file: {filepath}")

def run_cmd(cmd: list[str], cwd: Path | None = None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True)


def ensure_exists(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"{what} not found at {path}")
    print(f"[OK] {what}: {path}")


def pcs_pipeline(md_path: str | Path, repo_path: str | Path) -> bool:
    md_path = Path(md_path).resolve()
    repo_path = Path(repo_path).resolve()

    ensure_exists(md_path, "Paper Markdown")
    ensure_exists(repo_path, "Code repository")
    ensure_exists(CREATE_CHUNKS, "create_chunks.py")
    ensure_exists(CREATE_SYMBOLS, "create_symbols.py")
    ensure_exists(CREATE_MAP, "create_map.py")

    # Clear all previous output files
    delete_files_in_directory(DATA_DIR)

    print("=== Starting PCS Pipeline ===")

    # Step 1: Paper chunks
    print("\n--- Step 1: Build chunks ---")
    run_cmd([PY, str(CREATE_CHUNKS), str(md_path), str(CHUNKS_JSON)])
    ensure_exists(CHUNKS_JSON, "chunks.json")

    # Step 2: Repo symbols
    print("\n--- Step 2: Extract symbols ---")
    run_cmd([PY, str(CREATE_SYMBOLS), str(repo_path), "--out", str(SYMBOLS_JSON)])
    ensure_exists(SYMBOLS_JSON, "symbols.json")

    # Step 3: Build matches
    print("\n--- Step 3: Build matches ---")
    run_cmd([PY, str(CREATE_MAP), str(SYMBOLS_JSON), str(CHUNKS_JSON)], cwd=HERE)

    matches_here = (HERE / "matches.jsonl").resolve()
    ensure_exists(matches_here, "matches.jsonl")

    # Move final file to /data
    print(f"\n--- Finalize: Move matches.jsonl --> {MATCHES_JSON} ---")
    shutil.move(str(matches_here), str(MATCHES_JSON))
    ensure_exists(MATCHES_JSON, "matches.jsonl (final)")

    print("\n=== PCS Pipeline DONE ===")
    print(f"chunks : {CHUNKS_JSON}")
    print(f"symbols: {SYMBOLS_JSON}")
    print(f"matches: {MATCHES_JSON}")

    return True