from flask import (
    Flask, render_template, send_file, send_from_directory,
    abort, url_for, Response, request, redirect, flash
)
import os
import sys
import mimetypes
import pathlib
import uuid
from werkzeug.utils import secure_filename
sys.path.append("../Backend/papercodesync/src") 
from driver import pcs_pipeline


app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "ep2c-app"

MAX_VIEW_SIZE = 2_000_000
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".json", ".latex", ".md"}
LANGUAGES = ["Python", "Java", "C++", "JavaScript", "TypeScript"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# (DELETE WHEN FULL PIPELINE IS INTEGRATED)
PAPERCODESYNC_EXAMPLE = os.path.abspath(os.path.join(BASE_DIR, "../Backend/papercodesync/example"))
PAPER_MD   = os.path.join(PAPERCODESYNC_EXAMPLE, "paper.md")
PAPER_PATH = os.path.join(PAPERCODESYNC_EXAMPLE, "paper.pdf")
REPO_ROOT  = os.path.join(PAPERCODESYNC_EXAMPLE, "repo")

PAPERCODESYNC_DATA     = os.path.abspath(os.path.join(BASE_DIR, "../Backend/papercodesync/data"))
PAPERCODESYNC_SYMBOLS = os.path.join(PAPERCODESYNC_DATA, "symbols.json")
PAPERCODESYNC_CHUNKS  = os.path.join(PAPERCODESYNC_DATA, "chunks.json")
PAPERCODESYNC_MATCHES = os.path.join(PAPERCODESYNC_DATA, "matches.jsonl")

def _absnorm(p):
    return os.path.realpath(os.path.abspath(p))

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
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv"}]
        for f in files:
            full = os.path.join(root, f)
            if os.path.getsize(full) > 50_000_000:
                continue
            rel = os.path.relpath(full, REPO_ROOT).replace("\\", "/")
            paths.append(rel)
    paths.sort()
    return paths

def _allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", languages=LANGUAGES)

@app.route("/upload", methods=["POST"])
def upload():
    language = request.form.get("language", "").strip()
    file = request.files.get("paper")

    if not language or not file or file.filename == "":
        flash("Please upload a paper and choose a language.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    if not _allowed_file(filename) or file.mimetype not in ("application/pdf", "application/octet-stream"):
        flash("Only the following file types are allowed: PDF, LaTeX, JSON, MD.")
        return redirect(url_for("index"))

    unique_name = f"{uuid.uuid4().hex}.pdf"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(save_path)

    # === TODO: integrate full pipeline ===



    if pcs_pipeline is None:
        flash("Backend driver not available. Ensure pcs_pipeline is importable.")
        print("[EP2C] pcs_pipeline not importable.", flush=True)
        return redirect(url_for("index"))

    try:
        print("[EP2C] Running backend synchronization pipeline (skip demo)...", flush=True)
        # BLOCKING call, do not redirect until this returns
        pcs_pipeline(PAPER_MD, REPO_ROOT)
        print("[EP2C] Backend sync complete.", flush=True)
    except Exception as e:
        print(f"[ERROR] pcs_pipeline failed: {e}", flush=True)
        flash("Pipeline failed. Check server logs.")
        return redirect(url_for("index"))

    missing = [p for p in (PAPERCODESYNC_CHUNKS, PAPERCODESYNC_SYMBOLS, PAPERCODESYNC_MATCHES) if not os.path.exists(p)]
    if missing:
        print(f"[ERROR] Backend outputs missing: {missing}", flush=True)
        flash("Backend output missing — ensure PaperCodeSync generated chunks/symbols/matches.")
        return redirect(url_for("index"))


    return redirect(url_for("viewer", filename=unique_name, language=language))
    

@app.route("/viewer", methods=["GET"])
def viewer():
    filename = request.args.get("filename")
    language = request.args.get("language", "")

    files = [
        {
            "path": rel,
            "label": rel.split("/")[-1],
            "url": url_for("serve_code_file", subpath=rel),
        }
        for rel in _walk_repo()
    ]

    if filename:
        pdf_url = url_for("static", filename=f"uploads/{filename}")
        paper_path_for_header = os.path.join(UPLOAD_FOLDER, filename)
    else:
        pdf_url = url_for("serve_paper")
        paper_path_for_header = PAPER_PATH

    return render_template(
        "viewer.html",
        pdf_url=pdf_url,
        files=files,
        repo_root=REPO_ROOT,
        paper_path=paper_path_for_header,
        repo_ok=os.path.isdir(REPO_ROOT),
        paper_ok=os.path.isfile(paper_path_for_header),
        symbols_url=url_for("serve_symbols"),
        chunks_url=url_for("serve_chunks"),
        matches_url=url_for("serve_matches"),
        chosen_language=language,
    )

@app.route("/data/symbols.json")
def serve_symbols():
    if not os.path.isfile(PAPERCODESYNC_SYMBOLS): abort(404)
    return send_file(PAPERCODESYNC_SYMBOLS, mimetype="application/json")

@app.route("/data/chunks.json")
def serve_chunks():
    if not os.path.isfile(PAPERCODESYNC_CHUNKS): abort(404)
    return send_file(PAPERCODESYNC_CHUNKS, mimetype="application/json")

@app.route("/data/matches.jsonl")
def serve_matches():
    if not os.path.isfile(PAPERCODESYNC_MATCHES): abort(404)
    def generate():
        with open(PAPERCODESYNC_MATCHES, "rb") as f:
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
    print("EP2C running", flush=True)
    print(f"Paper (md): {PAPER_MD}", flush=True)
    print(f"Paper (pdf): {PAPER_PATH}", flush=True)
    print(f"Repo:  {REPO_ROOT}", flush=True)
    print(f"Backend data: {PAPERCODESYNC_DATA}", flush=True)
    app.run(debug=True, host="0.0.0.0", port=5001)
