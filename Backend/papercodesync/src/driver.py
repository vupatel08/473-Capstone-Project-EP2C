#!/usr/bin/env python3
"""PCS driver
Runs:
  1) MinerU parser: ../../mineru/out/paper/auto/paper.md
  2) create_chunks.py: ../app/chunks.json
  3) create_symbols.py: ../app/symbols.json
  4) create_map.py: matches.jsonl (then moves to ../app/matches.jsonl)
"""
import subprocess, sys, time, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent

PY = sys.executable 

PDF_PATH         = (HERE / "../app/paper.pdf").resolve()
REPO_PATH        = (HERE / "../app/repo").resolve()
APP_CHUNKS_JSON  = (HERE / "../app/chunks.json").resolve()
APP_SYMBOLS_JSON = (HERE / "../app/symbols.json").resolve()
APP_MATCHES_JSON = (HERE / "../app/matches.jsonl").resolve()

MINERU_PARSER    = (HERE / "../../mineru/parser.py").resolve()
MINERU_OUT_DIR   = (HERE / "../../mineru/out").resolve()
MINERU_MD_PATH   = (MINERU_OUT_DIR / "paper/auto/paper.md").resolve()

CREATE_CHUNKS    = (HERE / "create_chunks.py").resolve()
CREATE_SYMBOLS   = (HERE / "create_symbols.py").resolve()
CREATE_MAP       = (HERE / "create_map.py").resolve()

def run_cmd(cmd: list[str], cwd: Path | None = None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            check=True
        )
        return proc.returncode
    except subprocess.CalledProcessError as e:
        print(f"[ERR] Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def wait_for_file(path: Path, timeout_s: int = 60*30, poll_s: float = 1.0):
    print(f"Waiting for: {path}")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if path.exists():
            print(f"[OK] Found {path}")
            return
        time.sleep(poll_s)
    print(f"[ERR] Timeout waiting for {path}")
    sys.exit(1)

def ensure_exists(path: Path, what: str):
    if not path.exists():
        print(f"[ERR] {what} not found at {path}")
        sys.exit(1)
    print(f"[OK] {what}: {path}")

def main():
    print("=== EP2C / PCS DRIVER START ===")

    ensure_exists(PDF_PATH, "Input PDF")
    ensure_exists(REPO_PATH, "Repo directory")
    ensure_exists(MINERU_PARSER, "MinerU parser.py")
    ensure_exists(CREATE_CHUNKS, "create_chunks.py")
    ensure_exists(CREATE_SYMBOLS, "create_symbols.py")
    ensure_exists(CREATE_MAP, "create_map.py")

    MINERU_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Step 1: MinerU parse ---")
    run_cmd([PY, str(MINERU_PARSER), str(PDF_PATH), str(MINERU_OUT_DIR)])
    wait_for_file(MINERU_MD_PATH, timeout_s=60*30)

    print("\n--- Step 2: Build chunks ---")
    run_cmd([PY, str(CREATE_CHUNKS), str(MINERU_MD_PATH), str(APP_CHUNKS_JSON)])
    ensure_exists(APP_CHUNKS_JSON, "chunks.json")

    print("\n--- Step 3: Extract symbols ---")
    run_cmd([PY, str(CREATE_SYMBOLS), str(REPO_PATH), "--out", str(APP_SYMBOLS_JSON)])
    ensure_exists(APP_SYMBOLS_JSON, "symbols.json")

    print("\n--- Step 4: Build matches ---")
    run_cmd([PY, str(CREATE_MAP), str(APP_SYMBOLS_JSON), str(APP_CHUNKS_JSON)], cwd=HERE)

    matches_here = (HERE / "matches.jsonl").resolve()
    ensure_exists(matches_here, "matches.jsonl")

    print(f"\n--- Finalize: Move matches.jsonl --> {APP_MATCHES_JSON} ---")
    APP_MATCHES_JSON.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(matches_here), str(APP_MATCHES_JSON))
    ensure_exists(APP_MATCHES_JSON, "app/matches.jsonl")

    print("\n=== DONE ===")
    print(f"chunks : {APP_CHUNKS_JSON}")
    print(f"symbols: {APP_SYMBOLS_JSON}")
    print(f"matches: {APP_MATCHES_JSON}")

if __name__ == "__main__":
    main()
