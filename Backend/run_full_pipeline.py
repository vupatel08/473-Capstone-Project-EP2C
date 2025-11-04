#!/usr/bin/env python3
"""
EP2C Full Pipeline Orchestrator
Runs the complete pipeline: Planning → Analysis → Coding → Explanation Layer

This script orchestrates all phases of the EP2C pipeline:
1. Planning Phase - Creates overall plan, architecture design, logic design, config
2. Analysis Phase - Performs detailed logic analysis for each file
3. Coding Phase - Generates code files (includes explanation layer generation)
4. Summary - Provides final output summary
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Try loading from Backend directory first, then project root
backend_dir = Path(__file__).parent.resolve()
project_root = backend_dir.parent
env_paths = [
    backend_dir / ".env",
    project_root / ".env",
    backend_dir / ".env.example",  # Fallback to example for reference
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    # If no .env file found, try loading from current directory
    load_dotenv()


def run_full_pipeline(
    paper_pdf_path: str,
    paper_name: str,
    gpt_version: str = "o3-mini",
    paper_format: str = "JSON",
    output_base_dir: str = "outputs"
):
    """
    Run the complete EP2C pipeline.
    
    Args:
        paper_pdf_path: Path to input PDF paper (or JSON if paper_format="JSON")
        paper_name: Name identifier for the paper (e.g., "Transformer", "GAN")
        gpt_version: GPT model version to use (default: "o3-mini")
        paper_format: Paper format ("JSON" or "LaTeX")
        output_base_dir: Base directory for outputs (default: "outputs")
    """
    
    # Setup paths (backend_dir already defined above)
    output_dir = Path(output_base_dir) / "paper2code" / paper_name
    output_repo_dir = output_dir / f"{paper_name}_repo"
    
    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    output_repo_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup paths for paper input
    if paper_format == "JSON":
        pdf_json_path = paper_pdf_path
        pdf_latex_path = None
    else:
        pdf_json_path = None
        pdf_latex_path = paper_pdf_path
    
    print("\n" + "="*70)
    print("EP2C FULL PIPELINE")
    print("="*70)
    print(f"Paper Name:      {paper_name}")
    print(f"Paper Format:    {paper_format}")
    print(f"GPT Version:     {gpt_version}")
    print(f"Paper Input:     {paper_pdf_path}")
    print(f"Output Dir:      {output_dir}")
    print(f"Generated Repo:  {output_repo_dir}")
    print("="*70 + "\n")
    
    # Verify paper input exists
    if not os.path.exists(paper_pdf_path):
        print(f"❌ Error: Paper file not found: {paper_pdf_path}")
        sys.exit(1)
    
    # Step 1: Planning Phase
    print("\n" + "="*70)
    print("[1/3] PLANNING PHASE")
    print("="*70)
    print("Generating overall plan, architecture design, logic design, and config...\n")
    
    planning_cmd = [
        sys.executable,
        str(backend_dir / "models" / "1_planning.py"),
        "--paper_name", paper_name,
        "--gpt_version", gpt_version,
        "--paper_format", paper_format,
        "--output_dir", str(output_dir)
    ]
    if pdf_json_path:
        planning_cmd.extend(["--pdf_json_path", pdf_json_path])
    if pdf_latex_path:
        planning_cmd.extend(["--pdf_latex_path", pdf_latex_path])
    
    try:
        result = subprocess.run(planning_cmd, check=True, cwd=str(backend_dir))
        print("\n✅ Planning phase completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Planning phase failed with exit code {e.returncode}")
        sys.exit(1)
    
    # Step 2: Analysis Phase
    print("\n" + "="*70)
    print("[2/3] ANALYSIS PHASE")
    print("="*70)
    print("Performing detailed logic analysis for each file...\n")
    
    analysis_cmd = [
        sys.executable,
        str(backend_dir / "models" / "2_analyzing.py"),
        "--paper_name", paper_name,
        "--gpt_version", gpt_version,
        "--paper_format", paper_format,
        "--output_dir", str(output_dir)
    ]
    if pdf_json_path:
        analysis_cmd.extend(["--pdf_json_path", pdf_json_path])
    if pdf_latex_path:
        analysis_cmd.extend(["--pdf_latex_path", pdf_latex_path])
    
    try:
        result = subprocess.run(analysis_cmd, check=True, cwd=str(backend_dir))
        print("\n✅ Analysis phase completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Analysis phase failed with exit code {e.returncode}")
        sys.exit(1)
    
    # Step 3: Coding Phase (includes explanation layer generation)
    print("\n" + "="*70)
    print("[3/3] CODING PHASE")
    print("="*70)
    print("Generating code files and explanation layer...\n")
    
    coding_cmd = [
        sys.executable,
        str(backend_dir / "models" / "3_coding.py"),
        "--paper_name", paper_name,
        "--gpt_version", gpt_version,
        "--paper_format", paper_format,
        "--output_dir", str(output_dir),
        "--output_repo_dir", str(output_repo_dir)
    ]
    if pdf_json_path:
        coding_cmd.extend(["--pdf_json_path", pdf_json_path])
    if pdf_latex_path:
        coding_cmd.extend(["--pdf_latex_path", pdf_latex_path])
    
    try:
        result = subprocess.run(coding_cmd, check=True, cwd=str(backend_dir))
        print("\n✅ Coding phase completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Coding phase failed with exit code {e.returncode}")
        sys.exit(1)
    
    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\n📁 Output Directory:     {output_dir}")
    print(f"📦 Generated Repository: {output_repo_dir}")
    
    # Check for explanation layer
    explanation_dir = output_dir / "explanation_layer"
    if explanation_dir.exists():
        print(f"📚 Explanation Layer:     {explanation_dir}")
        print(f"   - Traceability Map:    {explanation_dir / 'traceability_map.json'}")
        print(f"   - README:              {explanation_dir / 'README.md'}")
        print(f"   - Missing Info:        {explanation_dir / 'missing_information.json'}")
        print(f"   - Metrics:             {explanation_dir / 'explainability_metrics.json'}")
    else:
        print(f"⚠️  Explanation Layer:    Not generated (check warnings above)")
    
    print("\n" + "="*70)
    print("✅ EP2C Pipeline Execution Complete!")
    print("="*70 + "\n")
    
    return output_dir, output_repo_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EP2C Full Pipeline - Generate code from research papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with JSON format paper
  python run_full_pipeline.py \\
      --paper_pdf outputs/paper.json \\
      --paper_name Transformer \\
      --gpt_version o3-mini

  # Run with LaTeX format paper
  python run_full_pipeline.py \\
      --paper_pdf paper.tex \\
      --paper_name GAN \\
      --paper_format LaTeX \\
      --gpt_version gpt-4

Environment Variables:
  OPENAI_API_KEY: Required - Your OpenAI API key
        """
    )
    
    parser.add_argument(
        "--paper_pdf",
        type=str,
        required=True,
        help="Path to paper PDF, JSON, or LaTeX file"
    )
    parser.add_argument(
        "--paper_name",
        type=str,
        required=True,
        help="Paper name identifier (e.g., 'Transformer', 'GAN')"
    )
    parser.add_argument(
        "--gpt_version",
        type=str,
        default="o3-mini",
        help="GPT model version to use (default: o3-mini)"
    )
    parser.add_argument(
        "--paper_format",
        type=str,
        default="JSON",
        choices=["JSON", "LaTeX"],
        help="Paper format (default: JSON)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Base output directory (default: outputs)"
    )
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        print("\n   Please create a .env file:")
        print("   1. Copy .env.example to .env:")
        print(f"      cp {backend_dir}/.env.example {backend_dir}/.env")
        print("   2. Edit .env and add your API key:")
        print("      OPENAI_API_KEY=sk-your-key-here")
        print("\n   Or set it as an environment variable:")
        print("      export OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Run pipeline
    run_full_pipeline(
        args.paper_pdf,
        args.paper_name,
        args.gpt_version,
        args.paper_format,
        args.output_dir
    )

