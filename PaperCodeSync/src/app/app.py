from flask import Flask, render_template, send_file, send_from_directory, abort, url_for, Response
import os
import mimetypes
import pathlib

app = Flask(__name__)
app.secret_key = "ep2c-demo"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHUNKS_PATH  = os.path.join(BASE_DIR, "chunks.json")
SYMBOLS_PATH = os.path.join(BASE_DIR, "symbols.json")
MATCHES_PATH = os.path.join(BASE_DIR, "matches.jsonl")
PAPER_PATH   = os.path.join(BASE_DIR, "paper.pdf")
REPO_ROOT    = os.path.join(BASE_DIR, "repo")

MAX_VIEW_SIZE = 2_000_000 

def _absnorm(p): return os.path.realpath(os.path.abspath(p))

def _is_within_repo(candidate_path):
    repo = pathlib.Path(_absnorm(REPO_ROOT))
    cand = pathlib.Path(_absnorm(candidate_path))
    try:
        cand.relative_to(repo)
        return True
    except ValueError:
        return False

def _walk_repo():
    paths = []
    if not os.path.isdir(REPO_ROOT):
        return paths
    for root, dirs, files in os.walk(REPO_ROOT):
        # prune noisy folders
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv"}]
        for f in files:
            full = os.path.join(root, f)
            if os.path.getsize(full) > 50_000_000:  # skip huge files
                continue
            rel = os.path.relpath(full, REPO_ROOT).replace("\\", "/")
            paths.append(rel)
    paths.sort()
    return paths


@app.route("/")
def viewer():
    files = [
        {
            "path": rel,
            "label": rel.split("/")[-1],
            "url": url_for("serve_code_file", subpath=rel),
        }
        for rel in _walk_repo()
    ]
    pdf_url = url_for("serve_paper")

    return render_template(
        "viewer.html",
        pdf_url=pdf_url,
        files=files,
        repo_root=REPO_ROOT,
        paper_path=PAPER_PATH,
        repo_ok=os.path.isdir(REPO_ROOT),
        paper_ok=os.path.isfile(PAPER_PATH),
        symbols_url=url_for("serve_symbols"),
        chunks_url=url_for("serve_chunks"),
        matches_url=url_for("serve_matches"),
    )

@app.route("/data/symbols.json")
def serve_symbols():
    if not os.path.isfile(SYMBOLS_PATH): abort(404)
    return send_file(SYMBOLS_PATH, mimetype="application/json")

@app.route("/data/chunks.json")
def serve_chunks():
    if not os.path.isfile(CHUNKS_PATH): abort(404)
    return send_file(CHUNKS_PATH, mimetype="application/json")

@app.route("/data/matches.jsonl")
def serve_matches():
    if not os.path.isfile(MATCHES_PATH): abort(404)
    def generate():
        with open(MATCHES_PATH, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk: break
                yield chunk
    return Response(generate(), mimetype="text/plain")

@app.route("/paper")
def serve_paper():
    if not os.path.isfile(PAPER_PATH):
        abort(404)
    return send_file(PAPER_PATH, mimetype="application/pdf", conditional=True)

@app.route("/code/<path:subpath>")
def serve_code_file(subpath):
    abs_target = _absnorm(os.path.join(REPO_ROOT, subpath))
    if not _is_within_repo(abs_target) or not os.path.isfile(abs_target):
        abort(404)
    if os.path.getsize(abs_target) > MAX_VIEW_SIZE:
        mt, _ = mimetypes.guess_type(abs_target)
        return send_from_directory(REPO_ROOT, subpath, mimetype=mt or "application/octet-stream")
    mt, _ = mimetypes.guess_type(abs_target)
    return send_from_directory(REPO_ROOT, subpath, mimetype=mt or "text/plain")

if __name__ == "__main__":
    print("EP2C running")
    print(f"Paper: {PAPER_PATH}")
    print(f"Repo:  {REPO_ROOT}")
    print(f"Data: {SYMBOLS_PATH}, {CHUNKS_PATH}, {MATCHES_PATH}")
    app.run(debug=True, host="0.0.0.0", port=5001)
