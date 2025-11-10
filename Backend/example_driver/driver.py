import sys, os, io, json, argparse, zipfile, requests
from pathlib import Path
from typing import List, Dict, Any, Iterable, Union
from dotenv import load_dotenv
# code gen LLM
from google import genai

HERE = Path(__file__).resolve().parent # Backend/example_driver/
ROOT = HERE.parent.parent # project root                                  
sys.path.append("../research-tracker")
from find_repo import get_repo_link
sys.path.append("../parsing")
from parser import parse_doc  


DOTENV_PATH = ROOT / ".env"                              
load_dotenv(DOTENV_PATH)
if not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError("GEMINI_API_KEY is not set.")

WORK_ROOT_DEFAULT = HERE                                   
GITHUB_DIR = "github_repo"
PARSE_DIR = "parse_output"                                 
GEN_REPO_DIR = "repo"                                     

# helper function for downloading a GitHub repo
def _download_github_repo(repo_url: str, extract_root: Path):
    extract_root = Path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    def _try(branch: str):
        url = repo_url.rstrip("/") + f"/archive/refs/heads/{branch}.zip"
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(extract_root)
        top = max(extract_root.iterdir(), key=lambda p: p.stat().st_mtime)
        return top

    out = _try("main") or _try("master")
    if not out:
        raise RuntimeError("Could not download ZIP for main/master branch.")
    return out.resolve()


FILE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "project_name": {"type": "string"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                }
            }
        }
    },
    "required": ["files"],
    "additionalProperties": True
}

def _safe_join(root: Path, rel: str):
    p = (root / rel).resolve()
    if not str(p).startswith(str(root.resolve())):
        raise ValueError(f"Illegal path outside repo root: {rel}")
    return p

def _write_repo_from_manifest(manifest: Dict[str, Any], target_dir: Path):
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    for f in manifest["files"]:
        rel = f["path"].lstrip("/").replace("\\", "/")
        if rel in ("", ".", ".."):
            continue
        dst = _safe_join(target_dir, rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(f["content"], encoding="utf-8")

    readme = target_dir / "README.md"
    if not readme.exists():
        title = manifest.get("project_name", "Generated Project")
        readme.write_text(f"# {title}\n", encoding="utf-8")

    return target_dir

# prompt for gemeni
PROMPT_HEADER = """
You are a repository generator. Produce a runnable, minimal yet complete codebase.
Follow these rules strictly:
- Output ONLY JSON matching the provided schema (no prose, no backticks).
- Create multiple files and folders that together run end-to-end.
- Include a README with quickstart steps.
- Prefer small, cohesive modules.
- No placeholders like 'TODO'; write working code.
- Keep all paths POSIX-style and relative.
- Make the code SUPER human-friendly: add abundant, high-quality comments in every source file
  to teach a beginner what's happening line-by-line and why. Treat every function as an opportunity
  to explain concepts from the paper in plain language with practical examples.
"""

def _read_text(p: Path, max_chars: int) -> str:
    text = p.read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        half = max_chars // 2
        text = text[:half] + "\n\n...[TRUNCATED]...\n\n" + text[-half:]
    return text

def _gather_context_sources(
    context: Union[Path, Iterable[Path]],
    max_chars_per_file: int = 120_000,
    max_files: int = 12
) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []

    if isinstance(context, Path) and context.is_dir():
        base = context
        for pat in ("**/*.md", "**/*.json"):
            for p in sorted(base.glob(pat)):
                try:
                    snippet = _read_text(p, max_chars_per_file)
                except Exception:
                    continue
                items.append({"path": str(p.relative_to(base)), "snippet": snippet})
                if len(items) >= max_files:
                    return items
        return items

    files = list(context) if not isinstance(context, Path) else [context]
    for p in files:
        p = Path(p)
        if not p.exists() or not p.is_file():
            continue
        try:
            snippet = _read_text(p, max_chars_per_file)
        except Exception:
            continue
        items.append({"path": p.name, "snippet": snippet})
        if len(items) >= max_files:
            break
    return items

def _gen_repo_with_gemini(target_dir: Path, context: Union[Path, Iterable[Path]], model: str = "gemini-2.5-pro") -> Path:
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    if any(target_dir.iterdir()):
        raise RuntimeError(f"Target directory is not empty: {target_dir}")

    client = genai.Client()  

    context_items = _gather_context_sources(context)

    contents = [
        {"role": "user", "parts": [{"text": PROMPT_HEADER}]},
        {"role": "user", "parts": [{"text": "Project context from parsed paper (MinerU outputs):"}]},
    ]

    for item in context_items:
        header = f"[FILE: {item['path']}]\n"
        contents.append({"role": "user", "parts": [{"text": header + item["snippet"]}]} )

    contents.append({
        "role": "user",
        "parts": [{
            "text": (
                "Generate a working repository implementing the paper's core idea. "
                "Prefer Python 3.11 with a clear entry point (e.g., src/main.py), "
                "minimal dependencies, and a README with install & run instructions."
            )
        }]
    })

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": FILE_SCHEMA,
        },
    )

    manifest = json.loads(resp.text)
    repo_path = _write_repo_from_manifest(manifest, target_dir)
    return repo_path.resolve()


def run(
    paper_pdf_path: str,
    work_root: str | Path = WORK_ROOT_DEFAULT,
    generated_repo_dir: str = GEN_REPO_DIR,
    model: str = "gemini-2.5-pro",
) -> str:   
    paper_pdf_path = str(Path(paper_pdf_path).resolve())
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    github_root = work_root / GITHUB_DIR
    parse_dir = work_root / PARSE_DIR
    out_repo_dir = work_root / generated_repo_dir

    print("Running EP2C Pipeline...", flush=True)
    print("Checking for existing GitHub repo...", flush=True)

    # check if there's already an existing github repo for this paper
    # repo_url = get_repo_link(paper_pdf_path)
    # if repo_url:
    #     print(f"Found GitHub repo: {repo_url}", flush=True)
    #     repo_dir = _download_github_repo(repo_url, github_root)
    #     return str(repo_dir)

    print("No existing GitHub repo found.", flush=True)
    print("Parsing paper for context...", flush=True)

    # parse via MinerU
    parse_dir.mkdir(parents=True, exist_ok=True)
    # parse_doc(paper_pdf_path, str(parse_dir), lang="en") 
    context = parse_dir

    print("Parsing complete. Generating repository with Gemini...", flush=True)

    # generate repo with Gemini 
    repo_dir = _gen_repo_with_gemini(out_repo_dir, context, model=model)

    print("Repository generation complete.", flush=True)
    return str(repo_dir)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True, help="Path to the paper PDF")
    ap.add_argument("--work_root", default=str(WORK_ROOT_DEFAULT),
                    help="Working directory (default: this folder)")
    ap.add_argument("--generated_repo_dir", default=GEN_REPO_DIR,
                    help="Folder name under work_root for Gemini output (default: 'repo')")
    ap.add_argument("--model", default="gemini-2.5-pro")
    args = ap.parse_args()

    final_path = run(
        paper_pdf_path=args.paper,
        work_root=args.work_root,
        generated_repo_dir=args.generated_repo_dir,
        model=args.model,
    )

    print(final_path)
