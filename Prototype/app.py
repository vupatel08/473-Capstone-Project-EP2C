from flask import Flask, render_template, request, redirect, url_for, flash
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "dev-secret"

# --- Config ---
# Use an absolute path for the upload folder and ensure it exists
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
ALLOWED_EXTENSIONS = {".pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # <<< IMPORTANT
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

LANGUAGES = ["Python", "JavaScript", "C++"]
COMPLEXITY = ["Basic", "Intermediate", "Full-featured"]

def allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", languages=LANGUAGES, complexities=COMPLEXITY)

@app.route("/upload", methods=["POST"])
def upload():
    language = request.form.get("language")
    complexity = request.form.get("complexity")
    file = request.files.get("paper")

    if not language or not complexity or not file or file.filename == "":
        flash("Please choose a PDF, language, and complexity.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    # Some browsers send application/octet-stream; we allow that fallback
    if not allowed_file(filename) or file.mimetype not in ("application/pdf", "application/octet-stream"):
        flash("Only PDF files are allowed.")
        return redirect(url_for("index"))

    # Save with a unique name and pass just the filename to the next page
    unique_name = f"{uuid.uuid4().hex}.pdf"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    return redirect(url_for("viewer",
                            pdf_filename=unique_name,
                            language=language,
                            complexity=complexity))

@app.route("/viewer", methods=["GET"])
def viewer():
    pdf_filename = request.args.get("pdf_filename")
    language = request.args.get("language")
    complexity = request.args.get("complexity")

    if not pdf_filename or not language or not complexity:
        flash("Missing inputs. Please upload again.")
        return redirect(url_for("index"))

    # Build the mock file tree based on the chosen language
    lang_key = language.lower().replace("+", "p").replace(" ", "")
    if "python" in lang_key:
        base = "static/sample_code/python"
        files = [
            {"path": "README.md", "label": "README.md"},
            {"path": "main.py", "label": "main.py"},
            {"path": "model.py", "label": "model.py"},
            {"path": "train.py", "label": "train.py"},
            {"path": "utils/data.py", "label": "utils/data.py"},
        ]
    elif "javascript" in lang_key:
        base = "static/sample_code/javascript"
        files = [
            {"path": "README.md", "label": "README.md"},
            {"path": "package.json", "label": "package.json"},
            {"path": "src/index.js", "label": "src/index.js"},
            {"path": "src/model.js", "label": "src/model.js"},
            {"path": "src/train.js", "label": "src/train.js"},
        ]
    else:
        base = "static/sample_code/cpp"
        files = [
            {"path": "README.md", "label": "README.md"},
            {"path": "CMakeLists.txt", "label": "CMakeLists.txt"},
            {"path": "src/main.cpp", "label": "src/main.cpp"},
            {"path": "src/model.cpp", "label": "src/model.cpp"},
            {"path": "include/model.hpp", "label": "include/model.hpp"},
        ]

    # Prepend static URLs for the code files
    for f in files:
        f["url"] = "/" + os.path.join(base, f["path"]).replace("\\", "/")

    # Build a proper static URL for the uploaded PDF
    pdf_url = url_for("static", filename=f"uploads/{pdf_filename}")

    return render_template("viewer.html",
                           pdf_url=pdf_url,
                           language=language,
                           complexity=complexity,
                           files=files)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
