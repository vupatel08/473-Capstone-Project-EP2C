#!/usr/bin/env python3
"""
EP2C Unified Pipeline Driver
Combines research tracker integration with full EP2C pipeline execution.

This unified driver:
1. Checks for existing GitHub repos via research tracker
2. Downloads repos if found, or runs full EP2C pipeline if not
3. Returns consistent results dictionary with all paths
4. Supports both CLI and programmatic usage
"""

import argparse
import os
import sys
import subprocess
import io
import zipfile
import requests
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
backend_dir = Path(__file__).parent.resolve()
project_root = backend_dir.parent
env_paths = [
    backend_dir / ".env",
    project_root / ".env",
    backend_dir / ".env.example",
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv()


def run_unified_pipeline(
    paper_pdf_path: str,
    paper_name: Optional[str] = None,
    gpt_version: str = "o3-mini",
    paper_format: str = "JSON",
    work_root: Optional[Union[str, Path]] = None,
    output_base_dir: str = "outputs",
    generated_repo_dir: str = "repo",
    paper_md_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Run the unified EP2C pipeline with research tracker integration.
    
    This function:
    1. Checks for existing GitHub repos via research tracker
    2. If found, downloads the repo and returns early
    3. If not found, automatically parses PDF files using MinerU (if input is PDF)
    4. Runs the full EP2C pipeline (Planning → Analysis → Coding → Explanation)
    5. Returns a consistent dictionary with all relevant paths
    
    Args:
        paper_pdf_path: Path to input paper PDF or already-parsed file (PDFs will be auto-parsed)
        paper_name: Name identifier for the paper (extracted from PDF if not provided)
        gpt_version: GPT model version to use (default: "o3-mini")
        paper_format: Paper format ("JSON" or "LaTeX") - auto-set to "LaTeX" if PDF is parsed
        work_root: Working directory root (default: Backend/example_driver)
        output_base_dir: Base directory for outputs (default: "outputs")
        generated_repo_dir: Directory name for generated repo (default: "repo")
        paper_md_path: Optional pre-parsed markdown file path (skips parsing if provided)
    
    Returns:
        Dictionary containing:
        - repo_path: Path to generated/downloaded repository
        - paper_md_path: Path to paper markdown (for PaperCodeSync)
        - paper_json_path: Path to paper JSON (if available)
        - output_dir: Main output directory
        - explanation_dir: Explanation layer directory
        - explanation_md_path: Path to EXPLANATION.md
        - planning_md_path: Path to PLANNING.md
        - analysis_md_path: Path to ANALYSIS.md
        - coding_md_path: Path to CODING.md
        - from_github: Whether repo came from GitHub (bool)
    """
    # Normalize paths
    paper_pdf_path = _normalize_path(paper_pdf_path)
    if work_root is None:
        # Default to example_driver directory
        work_root = backend_dir / "example_driver"
    work_root = _normalize_path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    
    # Extract paper name if not provided
    if paper_name is None:
        paper_name = paper_pdf_path.stem
    
    # Setup output directories
    output_dir = work_root / output_base_dir / "paper2code" / paper_name
    output_repo_dir = output_dir / f"{paper_name}_repo"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_repo_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup paths for paper input based on format
    if paper_format == "JSON":
        pdf_json_path = str(paper_pdf_path)
        pdf_latex_path = None
    else:
        pdf_json_path = None
        pdf_latex_path = str(paper_pdf_path) if not paper_md_path else paper_md_path
    
    # Use provided paper_md_path if available
    if paper_md_path:
        pdf_latex_path = str(_normalize_path(paper_md_path))
    
    print("\n" + "="*70)
    print("EP2C UNIFIED PIPELINE")
    print("="*70)
    print(f"Paper Name:      {paper_name}")
    print(f"Paper Format:    {paper_format}")
    print(f"GPT Version:     {gpt_version}")
    print(f"Paper Input:     {paper_pdf_path}")
    print(f"Output Dir:      {output_dir}")
    print(f"Generated Repo:  {output_repo_dir}")
    print("="*70 + "\n")
    
    # Verify paper input exists
    if not paper_pdf_path.exists() and not (paper_md_path and Path(paper_md_path).exists()):
        raise FileNotFoundError(f"Paper file not found: {paper_pdf_path}")
    
    # STEP 1: Check research tracker for existing GitHub repo
    print("Checking for existing GitHub repo...", flush=True)
    repo_url = _check_research_tracker(str(paper_pdf_path))
    
    if repo_url:
        # Found existing repo - download and return early
        print(f"Found GitHub repo: {repo_url}", flush=True)
        github_root = work_root / "github_repo"
        repo_dir = _download_github_repo(repo_url, github_root)
        
        # Try to find paper.md for PaperCodeSync
        paper_md = work_root / "parse_output" / "paper.md"
        
        return {
            "repo_path": str(repo_dir),
            "paper_md_path": str(paper_md) if paper_md.exists() else "",
            "paper_json_path": "",
            "output_dir": "",
            "explanation_dir": "",
            "explanation_md_path": "",
            "planning_md_path": "",
            "analysis_md_path": "",
            "coding_md_path": "",
            "from_github": True
        }
    
    print("No existing GitHub repo found. Running full EP2C pipeline...", flush=True)
    
    # STEP 2: Parse PDF if needed (convert PDF to markdown/JSON)
    pdf_json_path, pdf_latex_path, updated_format = _parse_paper_if_needed(
        paper_pdf_path=paper_pdf_path,
        paper_md_path=paper_md_path,
        paper_format=paper_format,
        work_root=work_root,
        paper_name=paper_name,
        output_dir=output_dir
    )
    
    # Update paper_format if PDF was parsed (PDFs become markdown, which uses "LaTeX" format)
    if updated_format:
        paper_format = updated_format
    
    # Get parse_output_dir if PDF was parsed (for accessing content_list.json with images)
    # Parse output is now in the same directory as the document's regular output
    parse_output_dir = output_dir / "parse_output" if (output_dir / "parse_output").exists() else None
    
    # STEP 3: Run full EP2C pipeline
    # Planning phase
    if not _run_planning_phase(paper_name, gpt_version, paper_format, output_dir, pdf_json_path, pdf_latex_path, parse_output_dir):
        raise RuntimeError("Planning phase failed")
    
    # Analysis phase
    if not _run_analysis_phase(paper_name, gpt_version, paper_format, output_dir, pdf_json_path, pdf_latex_path):
        raise RuntimeError("Analysis phase failed")
    
    # Coding phase (includes explanation layer)
    if not _run_coding_phase(paper_name, gpt_version, paper_format, output_dir, output_repo_dir, pdf_json_path, pdf_latex_path):
        raise RuntimeError("Coding phase failed")
    
    # STEP 4: Collect all paths for return dictionary
    explanation_dir = output_dir / "explanation_layer"
    explanation_md_path = explanation_dir / "EXPLANATION.md" if explanation_dir.exists() else None
    
    # Phase MD files
    planning_md_path = output_dir / "PLANNING.md"
    analysis_md_path = output_dir / "ANALYSIS.md"
    coding_md_path = output_dir / "CODING.md"
    
    # Paper markdown path (for PaperCodeSync)
    # Use parsed markdown if available, otherwise use provided paper_md_path or pdf_latex_path
    final_paper_md_path = pdf_latex_path if pdf_latex_path else (paper_md_path if paper_md_path else None)
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nOutput Directory:     {output_dir}")
    print(f"Generated Repository: {output_repo_dir}")
    
    if explanation_dir.exists():
        print(f"Explanation Layer:     {explanation_dir}")
        print(f"   - EXPLANATION.md:      {explanation_md_path}")
        print(f"   - Traceability Map:    {explanation_dir / 'traceability_map.json'}")
    
    print(f"Phase Documentation:")
    print(f"   - PLANNING.md:          {planning_md_path if planning_md_path.exists() else 'Not found'}")
    print(f"   - ANALYSIS.md:          {analysis_md_path if analysis_md_path.exists() else 'Not found'}")
    print(f"   - CODING.md:            {coding_md_path if coding_md_path.exists() else 'Not found'}")
    
    print("\n" + "="*70)
    print("EP2C Unified Pipeline Execution Complete!")
    print("="*70 + "\n")
    
    # Return unified result dictionary
    return {
        "repo_path": str(output_repo_dir),
        "paper_md_path": str(final_paper_md_path) if final_paper_md_path else "",
        "paper_json_path": str(pdf_json_path) if pdf_json_path else "",
        "output_dir": str(output_dir),
        "explanation_dir": str(explanation_dir) if explanation_dir.exists() else "",
        "explanation_md_path": str(explanation_md_path) if explanation_md_path and explanation_md_path.exists() else "",
        "planning_md_path": str(planning_md_path) if planning_md_path.exists() else "",
        "analysis_md_path": str(analysis_md_path) if analysis_md_path.exists() else "",
        "coding_md_path": str(coding_md_path) if coding_md_path.exists() else "",
        "from_github": False
    }


def _download_github_repo(repo_url: str, extract_root: Path) -> Path:
    """
    Download a GitHub repository as a ZIP and extract it.
    
    Args:
        repo_url: GitHub repository URL
        extract_root: Directory where the repo should be extracted
    
    Returns:
        Path to the extracted repository directory
    
    Raises:
        RuntimeError: If download fails for both main and master branches
    """
    extract_root = Path(extract_root).resolve()
    extract_root.mkdir(parents=True, exist_ok=True)

    def _try(branch: str) -> Optional[Path]:
        """Try to download a specific branch."""
        url = repo_url.rstrip("/") + f"/archive/refs/heads/{branch}.zip"
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                return None
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                z.extractall(extract_root)
            top = max(extract_root.iterdir(), key=lambda p: p.stat().st_mtime)
            return top
        except Exception:
            return None

    out = _try("main") or _try("master")
    if not out:
        raise RuntimeError(f"Could not download ZIP for main/master branch from {repo_url}")
    return out.resolve()


def _normalize_path(path: Union[str, Path]) -> Path:
    """Normalize a path string or Path object to a resolved Path."""
    return Path(path).resolve()


def _parse_paper_if_needed(
    paper_pdf_path: Path,
    paper_md_path: Optional[str],
    paper_format: str,
    work_root: Path,
    paper_name: str,
    output_dir: Path
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse PDF if input is a PDF file, otherwise use provided paths.
    
    This function checks if the input is a PDF file. If so, it uses MinerU parser
    to convert it to markdown format. If the file is already parsed or not a PDF,
    it returns the original paths.
    
    Args:
        paper_pdf_path: Path to the paper file (PDF or already parsed)
        paper_md_path: Optional pre-parsed markdown file path
        paper_format: Paper format ("JSON" or "LaTeX")
        work_root: Working directory root
        paper_name: Name identifier for the paper
        output_dir: Output directory where parse results should be placed
    
    Returns:
        Tuple of (pdf_json_path, pdf_latex_path, updated_format) where:
        - pdf_json_path: Path to JSON file (if JSON format)
        - pdf_latex_path: Path to markdown/LaTeX file (if LaTeX format)
        - updated_format: Updated format string if PDF was parsed (None if unchanged)
    """
    # If markdown/JSON already provided, use those
    if paper_md_path and Path(paper_md_path).exists():
        if paper_format == "JSON":
            return str(paper_md_path), None, None
        else:
            return None, str(paper_md_path), None
    
    # Check if input is a PDF file
    if paper_pdf_path.suffix.lower() == '.pdf':
        print("\n" + "="*70)
        print("PARSING PDF WITH MINERU")
        print("="*70)
        print(f"Input PDF: {paper_pdf_path}")
        print(f"Paper Name: {paper_name}\n")
        
        # Setup parse output directory - place it in the same directory as the document's regular output
        parse_output_dir = output_dir / "parse_output"
        parse_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import parser
        parsing_dir = backend_dir / "parsing"
        if str(parsing_dir) not in sys.path:
            sys.path.insert(0, str(parsing_dir))
        
        try:
            from parser import ep2c_parse
            
            # Parse PDF (default to English)
            print("Running MinerU parser...", flush=True)
            ep2c_parse(
                docs=[(str(paper_pdf_path), "en")],
                output_path=str(parse_output_dir)
            )
            
            # MinerU outputs to: <output_dir>/<pdf_stem>/auto/<pdf_stem>.md
            # Note: MinerU uses the PDF filename stem, not paper_name, for the directory structure
            pdf_stem = paper_pdf_path.stem
            
            # Try PDF stem directory first (MinerU's default behavior)
            parsed_output_base = parse_output_dir / pdf_stem / "auto"
            parsed_md_path = parsed_output_base / f"{pdf_stem}.md"
            
            # If that doesn't exist, try paper_name directory
            if not parsed_md_path.exists():
                parsed_output_base = parse_output_dir / paper_name / "auto"
                parsed_md_path = parsed_output_base / f"{pdf_stem}.md"
            
            # If still not found, try paper_name as filename
            if not parsed_md_path.exists():
                parsed_md_path = parsed_output_base / f"{paper_name}.md"
            
            # If still not found, look for any .md file in the auto directory
            if not parsed_md_path.exists():
                # Try PDF stem directory first
                if (parse_output_dir / pdf_stem / "auto").exists():
                    md_files = list((parse_output_dir / pdf_stem / "auto").glob("*.md"))
                    if md_files:
                        parsed_md_path = md_files[0]
                # Then try paper_name directory
                elif parsed_output_base.exists():
                    md_files = list(parsed_output_base.glob("*.md"))
                    if md_files:
                        parsed_md_path = md_files[0]
            
            if parsed_md_path.exists():
                print(f"✓ Parsed PDF → Markdown: {parsed_md_path}", flush=True)
                # Update paper_format to LaTeX since we now have markdown
                return None, str(parsed_md_path), "LaTeX"
            else:
                raise FileNotFoundError(
                    f"Parsed markdown not found. Expected at {parsed_md_path} or similar. "
                    f"Check parse_output directory: {parse_output_dir}"
                )
                
        except ImportError as e:
            print(f"Warning: Could not import parser: {e}", flush=True)
            print("Continuing with PDF path (may fail if models expect parsed format)", flush=True)
            # Fall back to original paths
            if paper_format == "JSON":
                return str(paper_pdf_path) if paper_pdf_path.exists() else None, None, None
            else:
                return None, str(paper_pdf_path) if paper_pdf_path.exists() else None, None
        except Exception as e:
            print(f"Error parsing PDF: {e}", flush=True)
            raise RuntimeError(f"Failed to parse PDF: {e}") from e
    
    # Not a PDF, return original paths based on format
    if paper_format == "JSON":
        return str(paper_pdf_path) if paper_pdf_path.exists() else None, None, None
    else:
        return None, str(paper_pdf_path) if paper_pdf_path.exists() else None, None


def _check_research_tracker(paper_pdf_path: str) -> Optional[str]:
    """
    Check research tracker for existing GitHub repository.
    
    Args:
        paper_pdf_path: Path to paper PDF file
    
    Returns:
        GitHub repository URL if found, None otherwise
    """
    # Add research-tracker to path
    research_tracker_dir = project_root / "Backend" / "research-tracker"
    if research_tracker_dir.exists() and str(research_tracker_dir) not in sys.path:
        sys.path.insert(0, str(research_tracker_dir))
    
    try:
        from find_repo import get_repo_link
        repo_url = get_repo_link(paper_pdf_path)
        return repo_url
    except ImportError:
        # Research tracker not available, continue without it
        return None
    except Exception as e:
        # Log error but don't fail - continue with pipeline
        print(f"Warning: Research tracker check failed: {e}", flush=True)
        return None


def _run_planning_phase(
    paper_name: str,
    gpt_version: str,
    paper_format: str,
    output_dir: Path,
    pdf_json_path: Optional[str],
    pdf_latex_path: Optional[str],
    parse_output_dir: Optional[Path] = None,
) -> bool:
    """
    Run the planning phase of the EP2C pipeline.
    
    Args:
        paper_name: Name identifier for the paper
        gpt_version: GPT model version to use
        paper_format: Paper format ("JSON" or "LaTeX")
        output_dir: Output directory for planning artifacts
        pdf_json_path: Path to paper JSON (if JSON format)
        pdf_latex_path: Path to paper LaTeX/Markdown (if LaTeX format)
        parse_output_dir: Optional path to parse_output directory containing content_list.json
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("[1/4] PLANNING PHASE")
    print("="*70)
    print("Generating overall plan, architecture design, logic design, and config...\n")
    
    # Build command for planning phase
    planning_cmd = [
        sys.executable,
        str(backend_dir / "models" / "1_planning.py"),
        "--paper_name", paper_name,
        "--gpt_version", gpt_version,
        "--paper_format", paper_format,
        "--output_dir", str(output_dir)
    ]
    
    # Add paper input path based on format
    if pdf_json_path:
        planning_cmd.extend(["--pdf_json_path", pdf_json_path])
    if pdf_latex_path:
        planning_cmd.extend(["--pdf_latex_path", pdf_latex_path])
    
    # Add parse_output_dir if available (for accessing content_list.json with images)
    if parse_output_dir and parse_output_dir.exists():
        planning_cmd.extend(["--parse_output_dir", str(parse_output_dir)])
    
    try:
        # Run planning phase subprocess
        result = subprocess.run(planning_cmd, check=True, cwd=str(backend_dir))
        print("\nPlanning phase completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nPlanning phase failed with exit code {e.returncode}")
        return False


def _run_analysis_phase(
    paper_name: str,
    gpt_version: str,
    paper_format: str,
    output_dir: Path,
    pdf_json_path: Optional[str],
    pdf_latex_path: Optional[str],
) -> bool:
    """
    Run the analysis phase of the EP2C pipeline.
    
    Args:
        paper_name: Name identifier for the paper
        gpt_version: GPT model version to use
        paper_format: Paper format ("JSON" or "LaTeX")
        output_dir: Output directory for analysis artifacts
        pdf_json_path: Path to paper JSON (if JSON format)
        pdf_latex_path: Path to paper LaTeX/Markdown (if LaTeX format)
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("[2/4] ANALYSIS PHASE")
    print("="*70)
    print("Performing detailed logic analysis for each file...\n")
    
    # Build command for analysis phase
    analysis_cmd = [
        sys.executable,
        str(backend_dir / "models" / "2_analyzing.py"),
        "--paper_name", paper_name,
        "--gpt_version", gpt_version,
        "--paper_format", paper_format,
        "--output_dir", str(output_dir)
    ]
    
    # Add paper input path based on format
    if pdf_json_path:
        analysis_cmd.extend(["--pdf_json_path", pdf_json_path])
    if pdf_latex_path:
        analysis_cmd.extend(["--pdf_latex_path", pdf_latex_path])
    
    try:
        # Run analysis phase subprocess
        result = subprocess.run(analysis_cmd, check=True, cwd=str(backend_dir))
        print("\nAnalysis phase completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nAnalysis phase failed with exit code {e.returncode}")
        return False


def _run_coding_phase(
    paper_name: str,
    gpt_version: str,
    paper_format: str,
    output_dir: Path,
    output_repo_dir: Path,
    pdf_json_path: Optional[str],
    pdf_latex_path: Optional[str],
) -> bool:
    """
    Run the coding phase of the EP2C pipeline (includes explanation layer).
    
    Args:
        paper_name: Name identifier for the paper
        gpt_version: GPT model version to use
        paper_format: Paper format ("JSON" or "LaTeX")
        output_dir: Output directory for coding artifacts
        output_repo_dir: Directory for generated code repository
        pdf_json_path: Path to paper JSON (if JSON format)
        pdf_latex_path: Path to paper LaTeX/Markdown (if LaTeX format)
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("[3/4] CODING PHASE")
    print("="*70)
    print("Generating code files...\n")
    
    # Build command for coding phase
    coding_cmd = [
        sys.executable,
        str(backend_dir / "models" / "3_coding.py"),
        "--paper_name", paper_name,
        "--gpt_version", gpt_version,
        "--paper_format", paper_format,
        "--output_dir", str(output_dir),
        "--output_repo_dir", str(output_repo_dir)
    ]
    
    # Add paper input path based on format
    if pdf_json_path:
        coding_cmd.extend(["--pdf_json_path", pdf_json_path])
    if pdf_latex_path:
        coding_cmd.extend(["--pdf_latex_path", pdf_latex_path])
    
    try:
        # Run coding phase subprocess (includes explanation layer generation)
        result = subprocess.run(coding_cmd, check=True, cwd=str(backend_dir))
        print("\nCoding phase completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nCoding phase failed with exit code {e.returncode}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EP2C Unified Pipeline - Generate code from research papers",
        epilog="Example: python unified_pipeline.py --paper_pdf paper.pdf --paper_name Transformer"
    )
    
    parser.add_argument("--paper_pdf", type=str, required=True, help="Path to paper PDF, JSON, or markdown file")
    parser.add_argument("--paper_name", type=str, default=None, help="Paper name (default: extracted from filename)")
    parser.add_argument("--gpt_version", type=str, default="o3-mini", help="GPT model version (default: o3-mini)")
    parser.add_argument("--paper_format", type=str, default="JSON", choices=["JSON", "LaTeX"], help="Paper format (default: JSON)")
    parser.add_argument("--work_root", type=str, default=None, help="Working directory root")
    parser.add_argument("--output_base_dir", type=str, default="outputs", help="Output base directory")
    parser.add_argument("--generated_repo_dir", type=str, default="repo", help="Generated repo directory name")
    parser.add_argument("--paper_md_path", type=str, default=None, help="Pre-parsed markdown file path")
    
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found. Set it with: export OPENAI_API_KEY=your_key")
        sys.exit(1)
    
    try:
        result = run_unified_pipeline(
            paper_pdf_path=args.paper_pdf,
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            paper_format=args.paper_format,
            work_root=args.work_root,
            output_base_dir=args.output_base_dir,
            generated_repo_dir=args.generated_repo_dir,
            paper_md_path=args.paper_md_path
        )
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Repository:  {result.get('repo_path', 'N/A')}")
        print(f"Output Dir:  {result.get('output_dir', 'N/A')}")
        print("="*70)
    
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

