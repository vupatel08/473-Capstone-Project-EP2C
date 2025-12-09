#!/usr/bin/env python3
"""
Script to run unified_pipeline.py for every PDF file in a folder.

Usage:
    python dataset_run.py <folder_path> [unified_pipeline_args...]

Example:
    # Basic usage - run unified_pipeline for all PDFs in folder
    python dataset_run.py ./papers

    # With additional arguments
    python dataset_run.py ./papers --gpt_version o3-mini --work_root ./work

    # Only process specific PDF pattern
    python dataset_run.py ./papers --pattern "*.pdf"

    # Recursively process PDFs in subdirectories
    python dataset_run.py ./papers --recursive 
"""

# python .\scripts\dataset_run.py <folder_path> --gpt_version gpt-4.1-nano --output_base_dir ../../dataset_out

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List


def run_unified_pipeline_for_pdfs(
    folder_path: Path,
    additional_args: List[str] = None,
    file_pattern: str = "*.pdf",
    recursive: bool = False,
    unified_pipeline_path: Path = None
) -> None:
    """
    Run unified_pipeline.py for every PDF file in a folder.
    
    Args:
        folder_path: Path to the folder containing PDF files
        additional_args: Additional arguments to pass to unified_pipeline.py
        file_pattern: Glob pattern to match files (default: "*.pdf")
        recursive: Whether to search subdirectories recursively
        unified_pipeline_path: Path to unified_pipeline.py (auto-detected if None)
    """
    # Auto-detect unified_pipeline.py location
    if unified_pipeline_path is None:
        # Assume this script is in Backend/scripts/, so unified_pipeline.py is in Backend/
        script_dir = Path(__file__).parent.resolve()
        unified_pipeline_path = script_dir.parent / "unified_pipeline.py"
    
    if not unified_pipeline_path.exists():
        print(f"❌ Error: unified_pipeline.py not found at: {unified_pipeline_path}")
        print(f"   Please specify the correct path with --unified_pipeline_path")
        sys.exit(1)
    
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"❌ Error: Folder not found or not a directory: {folder_path}")
        sys.exit(1)
    
    # Find all files matching the pattern
    if recursive:
        files = list(folder_path.rglob(file_pattern))
    else:
        files = list(folder_path.glob(file_pattern))
    
    # Filter to only files (not directories)
    files = [f for f in files if f.is_file()]
    
    if not files:
        print(f"⚠️  Warning: No files found in {folder_path} matching pattern '{file_pattern}'")
        return
    
    print(f"📁 Found {len(files)} PDF file(s) in {folder_path}")
    print(f"🚀 Running unified_pipeline.py for each PDF...\n")
    
    additional_args = additional_args or []
    success_count = 0
    fail_count = 0
    
    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing: {file_path.name}")
        
        # Extract paper name from filename (without extension)
        paper_name = file_path.stem
        
        # Build command: python unified_pipeline.py --paper_pdf file_path --paper_name paper_name [additional_args...]
        cmd = [
            sys.executable,
            str(unified_pipeline_path),
            "--paper_pdf", str(file_path),
            "--paper_name", paper_name
        ] + additional_args
        
        try:
            result = subprocess.run(
                cmd,
                check=False,  # Don't raise exception on non-zero exit
                capture_output=False,  # Show output in real-time
                cwd=unified_pipeline_path.parent  # Run from unified_pipeline.py's directory
            )
            
            if result.returncode == 0:
                success_count += 1
                print(f"✅ Success: {file_path.name}\n")
            else:
                fail_count += 1
                print(f"❌ Failed: {file_path.name} (exit code: {result.returncode})\n")
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user. Processed {i-1}/{len(files)} files.")
            sys.exit(1)
        except Exception as e:
            fail_count += 1
            print(f"❌ Error processing {file_path.name}: {e}\n")
    
    # Summary
    print("=" * 70)
    print(f"📊 Summary:")
    print(f"   Total files: {len(files)}")
    print(f"   ✅ Successful: {success_count}")
    print(f"   ❌ Failed: {fail_count}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run unified_pipeline.py for every PDF file in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run unified_pipeline for all PDFs in a folder
  python dataset_run.py ./papers

  # Run with additional arguments
  python dataset_run.py ./papers --gpt_version o3-mini --work_root ./work

  # Only process specific PDF pattern
  python dataset_run.py ./papers --pattern "*.pdf"

  # Recursively process PDFs in subdirectories
  python dataset_run.py ./papers --recursive

  # Specify custom unified_pipeline.py path
  python dataset_run.py ./papers --unified_pipeline_path ../unified_pipeline.py
        """
    )
    
    parser.add_argument(
        "folder_path",
        type=Path,
        help="Path to the folder containing PDF files to process"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="Glob pattern to match files (default: '*.pdf')"
    )
    
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search subdirectories recursively"
    )
    
    parser.add_argument(
        "--unified_pipeline_path",
        type=Path,
        default=None,
        help="Path to unified_pipeline.py (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "additional_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to unified_pipeline.py (e.g., --gpt_version, --work_root)"
    )
    
    args = parser.parse_args()
    
    run_unified_pipeline_for_pdfs(
        folder_path=args.folder_path,
        additional_args=args.additional_args,
        file_pattern=args.pattern,
        recursive=args.recursive,
        unified_pipeline_path=args.unified_pipeline_path
    )


if __name__ == "__main__":
    main()

