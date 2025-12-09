#!/usr/bin/env python3
"""
Paper2Code Batch Pipeline
Runs the unified pipeline on multiple papers from the Paper2Code dataset and evaluates results.

This script:
1. Loads papers from the Paper2Code dataset (or custom list)
2. Runs unified_pipeline.py on each paper
3. Optionally evaluates each generated repository
4. Generates comparison reports
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
backend_dir = Path(__file__).parent.parent.resolve()
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

# Add Backend to path for imports
sys.path.insert(0, str(backend_dir))

from unified_pipeline import run_unified_pipeline
from paper2code.eval import evaluate_from_pipeline


class Paper2CodeBatchPipeline:
    """Batch processing pipeline for Paper2Code dataset."""
    
    def __init__(
        self,
        work_root: Optional[Path] = None,
        output_base_dir: str = "outputs",
        gpt_version: str = "o3-mini",
        eval_gpt_version: str = "gpt-4o",
        skip_existing: bool = True
    ):
        """
        Initialize batch pipeline.
        
        Args:
            work_root: Working directory root
            output_base_dir: Base directory for outputs
            gpt_version: GPT version for code generation
            eval_gpt_version: GPT version for evaluation
            skip_existing: Skip papers that already have outputs
        """
        self.work_root = work_root or (backend_dir / "example_driver")
        self.output_base_dir = output_base_dir
        self.gpt_version = gpt_version
        self.eval_gpt_version = eval_gpt_version
        self.skip_existing = skip_existing
        
        # Load dataset info
        dataset_info_path = backend_dir / "paper2code" / "dataset_info.json"
        if dataset_info_path.exists():
            with open(dataset_info_path, 'r', encoding='utf-8') as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = {}
            print("⚠️  Warning: dataset_info.json not found. Will use custom paper list.")
    
    def get_papers_from_dataset(
        self,
        conferences: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get list of papers from the dataset.
        
        Args:
            conferences: List of conferences to include (e.g., ['iclr2024', 'nips2024'])
            limit: Maximum number of papers to process
        
        Returns:
            List of paper dictionaries with repo_name, repo_url, etc.
        """
        papers = []
        conferences = conferences or list(self.dataset_info.keys())
        
        for conference in conferences:
            if conference in self.dataset_info:
                papers.extend(self.dataset_info[conference])
        
        if limit:
            papers = papers[:limit]
        
        return papers
    
    def find_paper_file(self, paper_name: str, paper_data_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Find paper PDF or JSON file.
        
        Args:
            paper_name: Name of the paper
            paper_data_dir: Directory containing paper data files
        
        Returns:
            Path to paper file if found, None otherwise
        """
        if paper_data_dir is None:
            # Try common locations
            possible_dirs = [
                self.work_root / "papers",
                self.work_root.parent / "papers",
                Path("papers"),
            ]
            for dir_path in possible_dirs:
                if dir_path.exists():
                    paper_data_dir = dir_path
                    break
        
        if paper_data_dir and paper_data_dir.exists():
            # Try different file patterns
            patterns = [
                f"{paper_name}.pdf",
                f"{paper_name}.json",
                f"{paper_name}_cleaned.json",
            ]
            
            for pattern in patterns:
                file_path = paper_data_dir / pattern
                if file_path.exists():
                    return file_path
            
            # Try recursive search
            for pattern in patterns:
                matches = list(paper_data_dir.rglob(pattern))
                if matches:
                    return matches[0]
        
        return None
    
    def run_pipeline_for_paper(
        self,
        paper_info: Dict,
        paper_file: Optional[Path] = None,
        run_evaluation: bool = False,
        gold_repo_path: Optional[Path] = None
    ) -> Dict:
        """
        Run pipeline for a single paper.
        
        Args:
            paper_info: Paper information dictionary
            paper_file: Path to paper file (if None, will try to find it)
            run_evaluation: Whether to run evaluation after generation
            gold_repo_path: Path to gold repository for reference-based evaluation
        
        Returns:
            Result dictionary with status and paths
        """
        paper_name = paper_info.get("repo_name", paper_info.get("paper", "unknown"))
        
        print(f"\n{'='*70}")
        print(f"Processing: {paper_name}")
        print(f"{'='*70}")
        
        # Check if already exists
        output_dir = self.work_root / self.output_base_dir / "paper2code" / paper_name
        if self.skip_existing and output_dir.exists():
            print(f"⏭️  Skipping {paper_name} (output already exists)")
            return {
                "paper_name": paper_name,
                "status": "skipped",
                "output_dir": str(output_dir)
            }
        
        # Find paper file if not provided
        if paper_file is None:
            paper_file = self.find_paper_file(paper_name)
        
        if paper_file is None:
            print(f"❌ Paper file not found for {paper_name}")
            return {
                "paper_name": paper_name,
                "status": "error",
                "error": "Paper file not found"
            }
        
        print(f"📄 Paper file: {paper_file}")
        
        # Determine paper format
        paper_format = "LaTeX" if paper_file.suffix.lower() == '.pdf' else "JSON"
        
        try:
            # Run unified pipeline
            print(f"🚀 Running unified pipeline...")
            result = run_unified_pipeline(
                paper_pdf_path=str(paper_file),
                paper_name=paper_name,
                gpt_version=self.gpt_version,
                paper_format=paper_format,
                work_root=self.work_root,
                output_base_dir=self.output_base_dir
            )
            
            # Run evaluation if requested
            eval_result = None
            if run_evaluation:
                print(f"📊 Running evaluation...")
                try:
                    eval_result = evaluate_from_pipeline(
                        paper_path=result.get("paper_md_path") or result.get("paper_json_path"),
                        target_repo_path=result.get("repo_path"),
                        gold_repo_path=str(gold_repo_path) if gold_repo_path else None,
                        gpt_version=self.eval_gpt_version,
                        output_path=str(output_dir / "evaluation_results.json")
                    )
                    print(f"✅ Evaluation Score: {eval_result.get('score', 'N/A')}/5")
                except Exception as e:
                    print(f"⚠️  Evaluation failed: {e}")
                    eval_result = {"error": str(e)}
            
            return {
                "paper_name": paper_name,
                "status": "success",
                "output_dir": result.get("output_dir"),
                "repo_path": result.get("repo_path"),
                "evaluation": eval_result
            }
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            return {
                "paper_name": paper_name,
                "status": "error",
                "error": str(e)
            }
    
    def run_batch(
        self,
        papers: List[Dict],
        run_evaluation: bool = False,
        gold_repos_dir: Optional[Path] = None
    ) -> Dict:
        """
        Run pipeline on multiple papers.
        
        Args:
            papers: List of paper dictionaries
            run_evaluation: Whether to evaluate each generated repository
            gold_repos_dir: Directory containing gold repositories
        
        Returns:
            Summary dictionary with results
        """
        results = []
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        print(f"\n{'='*70}")
        print(f"BATCH PIPELINE - Processing {len(papers)} papers")
        print(f"{'='*70}\n")
        
        for i, paper_info in enumerate(papers, 1):
            paper_name = paper_info.get("repo_name", "unknown")
            print(f"\n[{i}/{len(papers)}] {paper_name}")
            
            # Find gold repo if available
            gold_repo_path = None
            if gold_repos_dir and gold_repos_dir.exists():
                # Try to find gold repo
                repo_name = paper_info.get("repo_name", "")
                possible_paths = [
                    gold_repos_dir / repo_name,
                    gold_repos_dir / f"{repo_name}-main",
                    gold_repos_dir / repo_name.replace("-", "_"),
                ]
                for path in possible_paths:
                    if path.exists():
                        gold_repo_path = path
                        break
            
            result = self.run_pipeline_for_paper(
                paper_info=paper_info,
                run_evaluation=run_evaluation,
                gold_repo_path=gold_repo_path
            )
            
            results.append(result)
            
            if result["status"] == "success":
                success_count += 1
            elif result["status"] == "error":
                error_count += 1
            elif result["status"] == "skipped":
                skipped_count += 1
        
        # Generate summary
        summary = {
            "total": len(papers),
            "success": success_count,
            "error": error_count,
            "skipped": skipped_count,
            "results": results
        }
        
        # Calculate average evaluation score if evaluations were run
        if run_evaluation:
            eval_scores = [
                r.get("evaluation", {}).get("score")
                for r in results
                if r.get("evaluation") and isinstance(r.get("evaluation"), dict) and "score" in r.get("evaluation", {})
            ]
            if eval_scores:
                summary["avg_evaluation_score"] = sum(eval_scores) / len(eval_scores)
                summary["evaluation_count"] = len(eval_scores)
        
        return summary
    
    def save_summary(self, summary: Dict, output_path: Path):
        """Save batch processing summary to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Summary saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run unified pipeline on Paper2Code dataset papers",
        epilog="""
Examples:
  # Run on all papers from dataset
  python batch_pipeline.py --conferences iclr2024 nips2024 --limit 10

  # Run with evaluation
  python batch_pipeline.py --conferences iclr2024 --run_evaluation

  # Run on custom paper list
  python batch_pipeline.py --papers paper1.json paper2.json

  # Run with gold repositories for reference-based evaluation
  python batch_pipeline.py --conferences iclr2024 --run_evaluation --gold_repos_dir ./gold_repos
        """
    )
    
    parser.add_argument(
        "--conferences",
        nargs="+",
        choices=["iclr2024", "icml2024", "nips2024"],
        help="Conferences to process papers from"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of papers to process"
    )
    
    parser.add_argument(
        "--papers",
        nargs="+",
        help="Custom list of paper JSON files to process"
    )
    
    parser.add_argument(
        "--work_root",
        type=Path,
        default=None,
        help="Working directory root (default: Backend/example_driver)"
    )
    
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="outputs",
        help="Output base directory (default: outputs)"
    )
    
    parser.add_argument(
        "--gpt_version",
        type=str,
        default="o3-mini",
        help="GPT version for code generation (default: o3-mini)"
    )
    
    parser.add_argument(
        "--eval_gpt_version",
        type=str,
        default="gpt-4o",
        help="GPT version for evaluation (default: gpt-4o)"
    )
    
    parser.add_argument(
        "--run_evaluation",
        action="store_true",
        help="Run evaluation on generated repositories"
    )
    
    parser.add_argument(
        "--gold_repos_dir",
        type=Path,
        help="Directory containing gold repositories for reference-based evaluation"
    )
    
    parser.add_argument(
        "--paper_data_dir",
        type=Path,
        help="Directory containing paper PDF/JSON files"
    )
    
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Don't skip papers that already have outputs"
    )
    
    parser.add_argument(
        "--output_summary",
        type=Path,
        help="Path to save batch processing summary JSON"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = Paper2CodeBatchPipeline(
        work_root=args.work_root,
        output_base_dir=args.output_base_dir,
        gpt_version=args.gpt_version,
        eval_gpt_version=args.eval_gpt_version,
        skip_existing=not args.no_skip_existing
    )
    
    # Get papers to process
    papers = []
    
    if args.papers:
        # Load custom paper list
        for paper_file in args.papers:
            if Path(paper_file).exists():
                with open(paper_file, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                    if isinstance(paper_data, list):
                        papers.extend(paper_data)
                    else:
                        papers.append(paper_data)
    elif args.conferences:
        # Get papers from dataset
        papers = pipeline.get_papers_from_dataset(
            conferences=args.conferences,
            limit=args.limit
        )
    else:
        print("❌ Error: Must specify either --conferences or --papers")
        sys.exit(1)
    
    if not papers:
        print("❌ Error: No papers to process")
        sys.exit(1)
    
    # Run batch processing
    summary = pipeline.run_batch(
        papers=papers,
        run_evaluation=args.run_evaluation,
        gold_repos_dir=args.gold_repos_dir
    )
    
    # Print summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total papers: {summary['total']}")
    print(f"✅ Successful: {summary['success']}")
    print(f"❌ Errors: {summary['error']}")
    print(f"⏭️  Skipped: {summary['skipped']}")
    
    if args.run_evaluation and 'avg_evaluation_score' in summary:
        print(f"\n📊 Evaluation Results:")
        print(f"   Average Score: {summary['avg_evaluation_score']:.2f}/5")
        print(f"   Evaluated: {summary['evaluation_count']}/{summary['success']}")
    
    print(f"{'='*70}\n")
    
    # Save summary
    if args.output_summary:
        pipeline.save_summary(summary, args.output_summary)
    else:
        # Save to default location
        summary_path = pipeline.work_root / args.output_base_dir / "paper2code" / "batch_summary.json"
        pipeline.save_summary(summary, summary_path)


if __name__ == "__main__":
    main()

