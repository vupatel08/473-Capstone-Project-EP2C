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
from typing import Dict, Optional, Union
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
    3. If not found, runs the full EP2C pipeline (Planning → Analysis → Coding → Explanation)
    4. Returns a consistent dictionary with all relevant paths
    
    Args:
        paper_pdf_path: Path to input paper PDF (used for research tracker check)
        paper_name: Name identifier for the paper (extracted from PDF if not provided)
        gpt_version: GPT model version to use (default: "o3-mini")
        paper_format: Paper format ("JSON" or "LaTeX")
        work_root: Working directory root (default: Backend/example_driver)
        output_base_dir: Base directory for outputs (default: "outputs")
        generated_repo_dir: Directory name for generated repo (default: "repo")
        paper_md_path: Optional pre-parsed markdown file path
    
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
    pass  # Implementation will be added in subsequent commits


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

