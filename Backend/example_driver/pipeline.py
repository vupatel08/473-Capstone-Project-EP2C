import sys, os, io, json, argparse, zipfile, requests
from pathlib import Path
from typing import List, Dict, Any, Iterable, Union
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent # Backend/example_driver/
ROOT = HERE.parent.parent # project root                                  
sys.path.append(str(ROOT / "Backend/research-tracker"))
from find_repo import get_repo_link

DOTENV_PATH = ROOT / ".env"                              
load_dotenv(DOTENV_PATH)

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
        {"role": "user", "parts": [{"text": "Project context from paper:"}]},
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
    gpt_version: str = "o3-mini",
    paper_name: str = None,
    paper_md_path: str = None,  # Optional: specify markdown file directly
) -> Dict[str, str]:
    """
    Run the EP2C pipeline using a hardcoded paper.md file.
    
    Args:
        paper_pdf_path: Path to paper PDF (used for research tracker check)
        work_root: Working directory root
        generated_repo_dir: Directory name for generated repo
        gpt_version: GPT model version (default: "o3-mini")
        paper_name: Name identifier for the paper (extracted from PDF if not provided)
    
    Returns:
        Dictionary containing paths to generated files
    """
    paper_pdf_path = str(Path(paper_pdf_path).resolve())
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    # Extract paper name if not provided
    if paper_name is None:
        paper_name = Path(paper_pdf_path).stem

    github_root = work_root / GITHUB_DIR
    parse_dir = work_root / PARSE_DIR

    print("Running EP2C Pipeline...", flush=True)
    print("Checking for existing GitHub repo...", flush=True)

    # Check if there's already an existing github repo for this paper (research tracker)
    try:
        repo_url = get_repo_link(paper_pdf_path)
        if repo_url:
            print(f"Found GitHub repo: {repo_url}", flush=True)
            repo_dir = _download_github_repo(repo_url, github_root)
            # Still try to find paper.md for PaperCodeSync
            paper_md_path = HERE / "parse_output" / "paper.md"
            return {
                "repo_path": str(repo_dir),
                "paper_md_path": str(paper_md_path) if paper_md_path.exists() else "",
                "paper_json_path": "",
                "output_dir": "",
                "explanation_dir": "",
                "from_github": True
            }
    except Exception as e:
        print(f"Research tracker check failed or no repo found: {e}", flush=True)

    print("No existing GitHub repo found.", flush=True)
    
    # Use provided paper_md_path or fall back to hardcoded paper.md
    if paper_md_path:
        paper_md_path = Path(paper_md_path).resolve()
        print(f"Using provided paper markdown: {paper_md_path}", flush=True)
    else:
        # Use hardcoded paper.md path (no MinerU parsing)
        paper_md_path = HERE / "parse_output" / "paper.md"
        print("Using hardcoded paper.md file...", flush=True)
    
    if not paper_md_path.exists():
        raise FileNotFoundError(
            f"Paper markdown file not found at {paper_md_path}. "
            f"Please ensure the markdown file exists, or provide paper_md_path parameter."
        )

    print(f"Found paper markdown at: {paper_md_path}", flush=True)

    # Import and call run_full_pipeline
    sys.path.append(str(ROOT / "Backend"))
    from run_full_pipeline import run_full_pipeline

    print("Running full EP2C pipeline (Planning → Analysis → Coding → Explanation)...", flush=True)

    # Run full pipeline with markdown as "LaTeX" format (both are just text)
    # Use absolute path since subprocess runs from Backend directory
    output_dir, output_repo_dir = run_full_pipeline(
        paper_pdf_path=str(paper_md_path.resolve()),  # Pass absolute path to markdown file
        paper_name=paper_name,
        gpt_version=gpt_version,
        paper_format="LaTeX",  # Treat markdown as LaTeX (both are plain text)
        output_base_dir=str(work_root / "outputs")
    )

    # Explanation layer is automatically generated in {output_dir}/explanation_layer/
    explanation_dir = output_dir / "explanation_layer"
    explanation_md_path = explanation_dir / "EXPLANATION.md"

    print("Pipeline complete!", flush=True)
    print(f"  Repository: {output_repo_dir}", flush=True)
    print(f"  Explanation: {explanation_md_path if explanation_md_path.exists() else explanation_dir}", flush=True)

    return {
        "repo_path": str(output_repo_dir),
        "paper_md_path": str(paper_md_path),  # For PaperCodeSync
        "paper_json_path": "",  # Not needed (using markdown directly)
        "output_dir": str(output_dir),
        "explanation_dir": str(explanation_dir),
        "explanation_md_path": str(explanation_md_path) if explanation_md_path.exists() else "",
        "from_github": False
    }



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True, help="Path to the paper PDF")
    ap.add_argument("--work_root", default=str(WORK_ROOT_DEFAULT),
                    help="Working directory (default: this folder)")
    ap.add_argument("--generated_repo_dir", default=GEN_REPO_DIR,
                    help="Folder name under work_root for generated output (default: 'repo')")
    ap.add_argument("--gpt_version", default="o3-mini", help="GPT model version (default: 'o3-mini')")
    ap.add_argument("--paper_name", help="Paper name identifier (extracted from PDF if not provided)")
    args = ap.parse_args()

    result = run(
        paper_pdf_path=args.paper,
        work_root=args.work_root,
        generated_repo_dir=args.generated_repo_dir,
        gpt_version=args.gpt_version,
        paper_name=args.paper_name,
    )

    print("\n" + "="*60)
    print("PIPELINE RESULTS")
    print("="*60)
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("="*60)
