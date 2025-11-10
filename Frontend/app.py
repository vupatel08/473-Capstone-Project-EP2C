from flask import (
    Flask, render_template, send_file, send_from_directory,
    abort, url_for, Response, request, redirect, flash, after_this_request
)
from pathlib import Path as PathLib
import tempfile
import zipfile
import os
import sys
import mimetypes
import pathlib
import uuid
from werkzeug.utils import secure_filename
sys.path.append("../Backend/papercodesync/src") 
from driver import pcs_pipeline
sys.path.append("../Backend/example_driver")
from pipeline import run as ep2c_pipeline


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
REPO_ROOT  = os.path.join(PAPERCODESYNC_EXAMPLE, "repo")
DRIVER_WORK_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../Backend/example_driver"))

PAPERCODESYNC_DATA     = os.path.abspath(os.path.join(BASE_DIR, "../Backend/papercodesync/data"))
PAPERCODESYNC_SYMBOLS = os.path.join(PAPERCODESYNC_DATA, "symbols.json")
PAPERCODESYNC_CHUNKS  = os.path.join(PAPERCODESYNC_DATA, "chunks.json")
PAPERCODESYNC_MATCHES = os.path.join(PAPERCODESYNC_DATA, "matches.jsonl")

EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", ".venv"}

def _zip_repo_to(temp_zip_path):
    with zipfile.ZipFile(temp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(REPO_ROOT):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                full = os.path.join(root, f)
                arc = os.path.relpath(full, REPO_ROOT)
                zf.write(full, arc)

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

@app.route("/export")
def export_repo():
    if not os.path.isdir(REPO_ROOT):
        abort(404)

    tmpdir = tempfile.mkdtemp(prefix="ep2c_zip_")
    zip_path = os.path.join(tmpdir, "repo.zip")
    _zip_repo_to(zip_path)

    @after_this_request
    def _cleanup(response):
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.isdir(tmpdir):
                os.rmdir(tmpdir)
        except Exception:
            pass
        return response

    download_name = f"{os.path.basename(REPO_ROOT.rstrip(os.sep)) or 'repo'}.zip"
    return send_file(zip_path, mimetype="application/zip", as_attachment=True, download_name=download_name)


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

    if ep2c_pipeline is None:
        flash("Backend driver not available. Ensure example_driver is importable.")
        print("[EP2C] example_driver not importable.", flush=True)
        return redirect(url_for("index"))
    
    try:
        result = ep2c_pipeline(
            paper_pdf_path=save_path,        
            work_root=DRIVER_WORK_ROOT,        
            generated_repo_dir="repo",
            gpt_version="o3-mini",  # Using OpenAI now, not Gemini
            paper_name=None,  # Will be extracted from PDF
        )

        # Update global REPO_ROOT so the rest of the app uses the new repo
        global REPO_ROOT
        REPO_ROOT = result["repo_path"]
        paper_md_path = result.get("paper_md_path", "")
        explanation_dir = result.get("explanation_dir", "")
        explanation_md_path = result.get("explanation_md_path", "")
        
        print(f"[EP2C] Driver produced repo at: {REPO_ROOT}", flush=True)
        print(f"[EP2C] Paper MD at: {paper_md_path}", flush=True)
        print(f"[EP2C] Explanation dir at: {explanation_dir}", flush=True)
        print(f"[EP2C] EXPLANATION.md at: {explanation_md_path}", flush=True)
    except Exception as e:
        print(f"[ERROR] driver_run failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        flash("Backend driver failed. Check server logs.")
        return redirect(url_for("index"))


    # Run PaperCodeSync if we have a paper MD path
    if paper_md_path and os.path.exists(paper_md_path):
        if pcs_pipeline is None:
            flash("Backend driver not available. Ensure pcs_pipeline is importable.")
            print("[EP2C] pcs_pipeline not importable.", flush=True)
            return redirect(url_for("index"))

        try:
            print("[EP2C] Running PaperCodeSync with parsed paper and generated repo...", flush=True)
            # BLOCKING call, do not redirect until this returns
            pcs_pipeline(paper_md_path, REPO_ROOT)
            print("[EP2C] PaperCodeSync complete.", flush=True)
        except Exception as e:
            print(f"[ERROR] pcs_pipeline failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            flash("PaperCodeSync failed. Check server logs.")
            return redirect(url_for("index"))
    else:
        print(f"[WARNING] Paper MD not found at {paper_md_path}, skipping PaperCodeSync", flush=True)

    # Check for PaperCodeSync files (optional - viewer works without them)
    missing = [p for p in (PAPERCODESYNC_CHUNKS, PAPERCODESYNC_SYMBOLS, PAPERCODESYNC_MATCHES) if not os.path.exists(p)]
    if missing:
        print(f"[WARNING] PaperCodeSync files missing: {missing}", flush=True)
        print("   Viewer will work but interactive mapping may not be available.", flush=True)
        # Don't redirect - allow viewer to work without PaperCodeSync


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
        flash("No paper specified. Please upload a PDF.")
        return redirect(url_for("index"))
    
    # Check if explanation layer exists - search for most recent EXPLANATION.md
    explanation_base_dir = os.path.join(DRIVER_WORK_ROOT, "outputs", "paper2code")
    explanation_md_path = None
    if os.path.exists(explanation_base_dir):
        explanation_files = []
        for root, dirs, files in os.walk(explanation_base_dir):
            if "EXPLANATION.md" in files:
                explanation_files.append(os.path.join(root, "EXPLANATION.md"))
        if explanation_files:
            # Use the most recent one
            explanation_md_path = max(explanation_files, key=os.path.getmtime)
        
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
        explanation_md_path=explanation_md_path,  # Pass to template
        chosen_language=language,
    )

@app.route("/data/explanation.md")
def serve_explanation():
    # Try to find EXPLANATION.md in the explanation layer
    explanation_base_dir = os.path.join(DRIVER_WORK_ROOT, "outputs", "paper2code")
    explanation_files = []
    for root, dirs, files in os.walk(explanation_base_dir):
        if "EXPLANATION.md" in files:
            explanation_files.append(os.path.join(root, "EXPLANATION.md"))
    
    if explanation_files:
        # Use the most recent one
        explanation_path = max(explanation_files, key=os.path.getmtime)
        return send_file(explanation_path, mimetype="text/markdown")
    else:
        abort(404)

@app.route("/data/symbols.json")
def serve_symbols():
    if not os.path.isfile(PAPERCODESYNC_SYMBOLS):
        return Response("[]", mimetype="application/json")
    return send_file(PAPERCODESYNC_SYMBOLS, mimetype="application/json")

@app.route("/data/chunks.json")
def serve_chunks():
    if not os.path.isfile(PAPERCODESYNC_CHUNKS):
        return Response("{}", mimetype="application/json")
    return send_file(PAPERCODESYNC_CHUNKS, mimetype="application/json")

@app.route("/data/matches.jsonl")
def serve_matches():
    if not os.path.isfile(PAPERCODESYNC_MATCHES):
        return Response("", mimetype="text/plain")
    def generate():
        with open(PAPERCODESYNC_MATCHES, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk: break
                yield chunk
    return Response(generate(), mimetype="text/plain")


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
    print(f"Repo:  {REPO_ROOT}", flush=True)
    print(f"Backend data: {PAPERCODESYNC_DATA}", flush=True)
    app.run(debug=True, host="0.0.0.0", port=5001)
