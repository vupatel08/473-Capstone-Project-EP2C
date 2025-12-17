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
sys.path.append("../Backend")
from unified_pipeline import run_unified_pipeline as ep2c_pipeline


app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "ep2c-app"

# Store REPO_ROOT in app config so it persists across Flask reloads
app.config['REPO_ROOT'] = None

MAX_VIEW_SIZE = 2_000_000
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".json", ".latex", ".md"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# (DELETE WHEN FULL PIPELINE IS INTEGRATED)
PAPERCODESYNC_EXAMPLE = os.path.abspath(os.path.join(BASE_DIR, "../Backend/papercodesync/example"))
PAPER_MD   = os.path.join(PAPERCODESYNC_EXAMPLE, "paper.md")
REPO_ROOT  = os.path.join(PAPERCODESYNC_EXAMPLE, "repo")
DRIVER_WORK_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../Backend/example_driver"))

# Initialize app config REPO_ROOT with the default
app.config['REPO_ROOT'] = REPO_ROOT

PAPERCODESYNC_DATA     = os.path.abspath(os.path.join(BASE_DIR, "../Backend/papercodesync/data"))
PAPERCODESYNC_SYMBOLS = os.path.join(PAPERCODESYNC_DATA, "symbols.json")
PAPERCODESYNC_CHUNKS  = os.path.join(PAPERCODESYNC_DATA, "chunks.json")
PAPERCODESYNC_MATCHES = os.path.join(PAPERCODESYNC_DATA, "matches.jsonl")

EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", ".venv"}

def _zip_repo_to(temp_zip_path):
    current_repo = app.config.get('REPO_ROOT', REPO_ROOT)
    with zipfile.ZipFile(temp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(current_repo):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                full = os.path.join(root, f)
                arc = os.path.relpath(full, current_repo)
                zf.write(full, arc)

def _absnorm(p):
    return os.path.realpath(os.path.abspath(p))

def _is_within_repo(candidate_path):
    current_repo = app.config.get('REPO_ROOT', REPO_ROOT)
    repo = pathlib.Path(_absnorm(current_repo))
    cand = pathlib.Path(_absnorm(candidate_path))
    try:
        cand.relative_to(repo)
        return True
    except ValueError:
        return False

def _walk_repo():
    paths = []
    current_repo = app.config.get('REPO_ROOT', REPO_ROOT)
    repo_root_norm = _absnorm(current_repo)
    print(f"[EP2C] _walk_repo() REPO_ROOT={current_repo} normalized={repo_root_norm}", flush=True)
    if not os.path.isdir(repo_root_norm):
        print(f"[EP2C] _walk_repo: not a directory", flush=True)
        return paths
    for root, dirs, files in os.walk(repo_root_norm):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv"}]
        for f in files:
            full = os.path.join(root, f)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            if size > 50_000_000:
                continue
            if not os.path.isfile(full) or not os.access(full, os.R_OK):
                continue
            rel = os.path.relpath(full, repo_root_norm).replace("\\", "/")
            paths.append(rel)
    paths.sort()
    print(f"[EP2C] _walk_repo() found {len(paths)} files", flush=True)
    if paths:
        print(f"[EP2C] _walk_repo() first 5 files: {paths[:5]}", flush=True)
    return paths

def _allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/export")
def export_repo():
    current_repo = app.config.get('REPO_ROOT', REPO_ROOT)
    if not os.path.isdir(current_repo):
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

    download_name = f"{os.path.basename(current_repo.rstrip(os.sep)) or 'repo'}.zip"
    return send_file(zip_path, mimetype="application/zip", as_attachment=True, download_name=download_name)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("paper")

    if not file or file.filename == "":
        flash("Please upload a paper.")
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
            gpt_version="gpt-4.1-nano",  # Using OpenAI now, not Gemini
            paper_name=None,  # Will be extracted from PDF
        )

        # Update app.config REPO_ROOT so it persists across Flask reloads
        old_root = app.config.get('REPO_ROOT', REPO_ROOT)
        app.config['REPO_ROOT'] = result["repo_path"]
        paper_md_path = result.get("paper_md_path", "")
        explanation_dir = result.get("explanation_dir", "")
        explanation_md_path = result.get("explanation_md_path", "")
        
        print(f"[EP2C] ===== UPLOAD: REPO_ROOT UPDATED =====", flush=True)
        print(f"[EP2C] OLD REPO_ROOT: {old_root}", flush=True)
        print(f"[EP2C] NEW REPO_ROOT: {app.config['REPO_ROOT']}", flush=True)
        print(f"[EP2C] Driver produced repo at: {app.config['REPO_ROOT']}", flush=True)
        print(f"[EP2C] repo_path exists={os.path.exists(app.config['REPO_ROOT'])} isdir={os.path.isdir(app.config['REPO_ROOT'])}", flush=True)
        if os.path.isdir(app.config['REPO_ROOT']):
            files_in_repo = os.listdir(app.config['REPO_ROOT'])
            print(f"[EP2C] files in repo dir: {files_in_repo}", flush=True)
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
            print(f"[EP2C] PCS: paper_md_path={paper_md_path} REPO_ROOT={app.config['REPO_ROOT']}", flush=True)
            # BLOCKING call, do not redirect until this returns
            pcs_pipeline(paper_md_path, app.config['REPO_ROOT'])
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


    return redirect(url_for("viewer", filename=unique_name))
    

@app.route("/viewer", methods=["GET"])
def viewer():
    print(f"\n[EP2C] ===== VIEWER ROUTE START =====", flush=True)
    filename = request.args.get("filename")
    print(f"[EP2C] viewer() received filename={filename}", flush=True)

    # CRITICAL: Log REPO_ROOT state at viewer entry
    current_repo = app.config.get('REPO_ROOT', REPO_ROOT)
    print(f"[EP2C] viewer: REPO_ROOT at entry={current_repo}", flush=True)
    print(f"[EP2C] viewer: REPO_ROOT exists={os.path.exists(current_repo)}", flush=True)
    print(f"[EP2C] viewer: REPO_ROOT isdir={os.path.isdir(current_repo)}", flush=True)
    
    files = [
        {
            "path": rel,
            "label": rel.split("/")[-1],
            "url": url_for("serve_code_file", subpath=rel),
        }
        for rel in _walk_repo()
    ]

    # Debug: log what repository root and files are being used to render the viewer
    print(f"[EP2C] viewer files_list={len(files)} items", flush=True)
    try:
        sample = [f["path"] for f in files][:5]
        print(f"[EP2C] viewer files first 5: {sample}", flush=True)
        if len(files) == 0:
            print(f"[EP2C] WARNING: No files found! _walk_repo() returned empty list", flush=True)
    except Exception as ex:
        print(f"[EP2C] viewer debug: could not summarize files: {ex}", flush=True)

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
        for root, dirs, walk_files in os.walk(explanation_base_dir):
            if "EXPLANATION.md" in walk_files:
                explanation_files.append(os.path.join(root, "EXPLANATION.md"))
        if explanation_files:
            # Use the most recent one
            explanation_md_path = max(explanation_files, key=os.path.getmtime)
        
    print(f"[EP2C] viewer: passing {len(files)} files to template", flush=True)
    print(f"[EP2C] viewer: files list being passed: {[f['path'] for f in files[:5]]}", flush=True)
    print(f"[EP2C] ===== VIEWER ROUTE END (rendering template) =====", flush=True)
    
    return render_template(
        "viewer.html",
        pdf_url=pdf_url,
        files=files,
        repo_root=current_repo,
        paper_path=paper_path_for_header,
        repo_ok=os.path.isdir(current_repo),
        paper_ok=os.path.isfile(paper_path_for_header),
        symbols_url=url_for("serve_symbols"),
        chunks_url=url_for("serve_chunks"),
        matches_url=url_for("serve_matches"),
        explanation_md_path=explanation_md_path,  # Pass to template
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

@app.route("/data/planning.md")
def serve_planning():
    """Serve PLANNING.md from the most recent pipeline run."""
    planning_base_dir = os.path.join(DRIVER_WORK_ROOT, "outputs", "paper2code")
    planning_files = []
    for root, dirs, files in os.walk(planning_base_dir):
        if "PLANNING.md" in files:
            planning_files.append(os.path.join(root, "PLANNING.md"))
    
    if planning_files:
        # Use the most recent one
        planning_path = max(planning_files, key=os.path.getmtime)
        return send_file(planning_path, mimetype="text/markdown")
    else:
        abort(404)

@app.route("/data/analysis.md")
def serve_analysis():
    """Serve ANALYSIS.md from the most recent pipeline run."""
    analysis_base_dir = os.path.join(DRIVER_WORK_ROOT, "outputs", "paper2code")
    analysis_files = []
    for root, dirs, files in os.walk(analysis_base_dir):
        if "ANALYSIS.md" in files:
            analysis_files.append(os.path.join(root, "ANALYSIS.md"))
    
    if analysis_files:
        # Use the most recent one
        analysis_path = max(analysis_files, key=os.path.getmtime)
        return send_file(analysis_path, mimetype="text/markdown")
    else:
        abort(404)

@app.route("/data/coding.md")
def serve_coding():
    """Serve CODING.md from the most recent pipeline run."""
    coding_base_dir = os.path.join(DRIVER_WORK_ROOT, "outputs", "paper2code")
    coding_files = []
    for root, dirs, files in os.walk(coding_base_dir):
        if "CODING.md" in files:
            coding_files.append(os.path.join(root, "CODING.md"))
    
    if coding_files:
        # Use the most recent one
        coding_path = max(coding_files, key=os.path.getmtime)
        return send_file(coding_path, mimetype="text/markdown")
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


@app.route("/debug/files")
def debug_files():
    """Return JSON describing the current REPO_ROOT and files found there.

    Useful for debugging why the viewer does or does not display repo files.
    """
    from flask import jsonify
    current_repo = app.config.get('REPO_ROOT', REPO_ROOT)
    print(f"[EP2C] /debug/files called, REPO_ROOT={current_repo}", flush=True)
    file_list = _walk_repo()
    checks = []
    for rel in file_list[:200]:
        abs_path = os.path.join(current_repo, rel)
        exists = os.path.isfile(abs_path)
        readable = os.access(abs_path, os.R_OK)
        size = os.path.getsize(abs_path) if exists else None
        checks.append({"path": rel, "exists": exists, "readable": readable, "size": size})

    return jsonify({
        "repo_root": current_repo,
        "files_found": len(file_list),
        "sample": checks,
    })


@app.route("/code/<path:subpath>")
def serve_code_file(subpath):
    current_repo = app.config.get('REPO_ROOT', REPO_ROOT)
    abs_target = _absnorm(os.path.join(current_repo, subpath))
    if not _is_within_repo(abs_target) or not os.path.isfile(abs_target):
        abort(404)
    if os.path.getsize(abs_target) > MAX_VIEW_SIZE:
        mt, _ = mimetypes.guess_type(abs_target)
        return send_from_directory(current_repo, subpath, mimetype=mt or "application/octet-stream")
    mt, _ = mimetypes.guess_type(abs_target)
    return send_from_directory(current_repo, subpath, mimetype=mt or "text/plain")

if __name__ == "__main__":
    print("EP2C running", flush=True)
    print(f"Paper (md): {PAPER_MD}", flush=True)
    print(f"Repo:  {REPO_ROOT}", flush=True)
    print(f"Backend data: {PAPERCODESYNC_DATA}", flush=True)
    app.run(debug=True, host="0.0.0.0", port=5001)
