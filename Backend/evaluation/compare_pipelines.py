#!/usr/bin/env python3
"""
Pipeline Comparison Script
Compares EP2C unified pipeline results with Paper2Code gold repositories.

This script:
1. Finds papers processed by the unified pipeline
2. Compares generated repositories with gold repositories from Paper2Code dataset
3. Evaluates both using the evaluation script
4. Generates comparison reports
"""

import argparse
import json
import sys
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

from paper2code.eval import evaluate_from_pipeline


class PipelineComparator:
    """Compare EP2C pipeline results with Paper2Code gold repositories."""
    
    def __init__(
        self,
        ep2c_output_dir: Path,
        gold_repos_dir: Optional[Path] = None,
        eval_gpt_version: str = "gpt-4o"
    ):
        """
        Initialize comparator.
        
        Args:
            ep2c_output_dir: Directory containing EP2C pipeline outputs
            gold_repos_dir: Directory containing gold repositories (from Paper2Code dataset)
            eval_gpt_version: GPT version for evaluation
        """
        self.ep2c_output_dir = Path(ep2c_output_dir)
        self.gold_repos_dir = Path(gold_repos_dir) if gold_repos_dir else None
        self.eval_gpt_version = eval_gpt_version
        
        # Load dataset info to map paper names to repo URLs
        dataset_info_path = backend_dir / "paper2code" / "dataset_info.json"
        if dataset_info_path.exists():
            with open(dataset_info_path, 'r', encoding='utf-8') as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = {}
            print("⚠️  Warning: dataset_info.json not found")
    
    def find_ep2c_papers(self) -> List[Dict]:
        """
        Find all papers processed by EP2C pipeline.
        
        Returns:
            List of paper info dictionaries
        """
        papers = []
        
        if not self.ep2c_output_dir.exists():
            print(f"❌ EP2C output directory not found: {self.ep2c_output_dir}")
            return papers
        
        # Look for paper directories in output_dir/paper2code/
        paper2code_dir = self.ep2c_output_dir / "paper2code"
        if not paper2code_dir.exists():
            paper2code_dir = self.ep2c_output_dir
        
        for paper_dir in paper2code_dir.iterdir():
            if not paper_dir.is_dir():
                continue
            
            paper_name = paper_dir.name
            
            # Check if this paper has a generated repository
            repo_dir = paper_dir / f"{paper_name}_repo"
            if not repo_dir.exists():
                continue
            
            # Find paper markdown/JSON
            paper_md = None
            parse_output_dir = paper_dir / "parse_output" / paper_name / "auto"
            if parse_output_dir.exists():
                md_files = list(parse_output_dir.glob("*.md"))
                if md_files:
                    paper_md = md_files[0]
            
            if not paper_md:
                # Try alternative locations
                alt_paths = [
                    paper_dir / "parse_output" / "paper.md",
                    paper_dir / f"{paper_name}.md",
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        paper_md = alt_path
                        break
            
            papers.append({
                "paper_name": paper_name,
                "ep2c_repo_path": str(repo_dir),
                "paper_path": str(paper_md) if paper_md else None,
                "output_dir": str(paper_dir)
            })
        
        return papers
    
    def find_gold_repo(self, paper_name: str) -> Optional[Path]:
        """
        Find gold repository for a paper.
        
        Args:
            paper_name: Name of the paper
            
        Returns:
            Path to gold repository if found, None otherwise
        """
        if not self.gold_repos_dir or not self.gold_repos_dir.exists():
            return None
        
        # Try different naming conventions
        possible_names = [
            paper_name,
            f"{paper_name}-main",
            paper_name.replace("-", "_"),
            paper_name.replace("_", "-"),
        ]
        
        for name in possible_names:
            repo_path = self.gold_repos_dir / name
            if repo_path.exists() and repo_path.is_dir():
                return repo_path
        
        # Try recursive search
        for name in possible_names:
            matches = list(self.gold_repos_dir.rglob(name))
            if matches:
                return matches[0]
        
        return None
    
    def get_repo_url_from_dataset(self, paper_name: str) -> Optional[str]:
        """Get GitHub repository URL from dataset info."""
        for conference, papers in self.dataset_info.items():
            for paper_info in papers:
                if paper_info.get("repo_name") == paper_name:
                    return paper_info.get("repo_url")
        return None
    
    def compare_paper(
        self,
        paper_info: Dict,
        compare_with_gold: bool = True,
        run_evaluation: bool = True
    ) -> Dict:
        """
        Compare EP2C results for a single paper.
        
        Args:
            paper_info: Paper information dictionary
            compare_with_gold: Whether to compare with gold repository
            run_evaluation: Whether to run evaluation
        
        Returns:
            Comparison result dictionary
        """
        paper_name = paper_info["paper_name"]
        ep2c_repo_path = paper_info["ep2c_repo_path"]
        paper_path = paper_info.get("paper_path")
        
        print(f"\n{'='*70}")
        print(f"Comparing: {paper_name}")
        print(f"{'='*70}")
        
        result = {
            "paper_name": paper_name,
            "ep2c_repo_path": ep2c_repo_path,
            "paper_path": paper_path,
        }
        
        # Find gold repository
        gold_repo_path = None
        if compare_with_gold:
            gold_repo_path = self.find_gold_repo(paper_name)
            if gold_repo_path:
                result["gold_repo_path"] = str(gold_repo_path)
                print(f"✅ Found gold repository: {gold_repo_path}")
            else:
                repo_url = self.get_repo_url_from_dataset(paper_name)
                if repo_url:
                    result["gold_repo_url"] = repo_url
                    print(f"⚠️  Gold repository not found locally, but URL available: {repo_url}")
                else:
                    print(f"⚠️  Gold repository not found for {paper_name}")
        
        # Run evaluations
        if run_evaluation and paper_path:
            # Evaluate EP2C repository
            print(f"\n📊 Evaluating EP2C repository...")
            try:
                ep2c_eval = evaluate_from_pipeline(
                    paper_path=paper_path,
                    target_repo_path=ep2c_repo_path,
                    gold_repo_path=str(gold_repo_path) if gold_repo_path else None,
                    gpt_version=self.eval_gpt_version,
                    output_path=str(Path(paper_info["output_dir"]) / "ep2c_evaluation.json")
                )
                result["ep2c_evaluation"] = ep2c_eval
                print(f"   EP2C Score: {ep2c_eval.get('score', 'N/A')}/5")
            except Exception as e:
                print(f"   ❌ EP2C evaluation failed: {e}")
                result["ep2c_evaluation"] = {"error": str(e)}
            
            # Evaluate gold repository (reference-free) if available
            if gold_repo_path:
                print(f"\n📊 Evaluating gold repository...")
                try:
                    gold_eval = evaluate_from_pipeline(
                        paper_path=paper_path,
                        target_repo_path=str(gold_repo_path),
                        gpt_version=self.eval_gpt_version,
                        output_path=str(Path(paper_info["output_dir"]) / "gold_evaluation.json")
                    )
                    result["gold_evaluation"] = gold_eval
                    print(f"   Gold Score: {gold_eval.get('score', 'N/A')}/5")
                    
                    # Calculate score difference
                    ep2c_score = result.get("ep2c_evaluation", {}).get("score")
                    gold_score = gold_eval.get("score")
                    if ep2c_score is not None and gold_score is not None:
                        result["score_difference"] = ep2c_score - gold_score
                        result["score_ratio"] = ep2c_score / gold_score if gold_score > 0 else None
                        print(f"   Score Difference: {result['score_difference']:+.2f}")
                        print(f"   Score Ratio: {result['score_ratio']:.2%}" if result['score_ratio'] else "   Score Ratio: N/A")
                except Exception as e:
                    print(f"   ❌ Gold evaluation failed: {e}")
                    result["gold_evaluation"] = {"error": str(e)}
        
        return result
    
    def compare_all(
        self,
        compare_with_gold: bool = True,
        run_evaluation: bool = True,
        paper_filter: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare all EP2C papers.
        
        Args:
            compare_with_gold: Whether to compare with gold repositories
            run_evaluation: Whether to run evaluation
            paper_filter: Optional list of paper names to filter
        
        Returns:
            Comparison summary dictionary
        """
        papers = self.find_ep2c_papers()
        
        if paper_filter:
            papers = [p for p in papers if p["paper_name"] in paper_filter]
        
        if not papers:
            print("❌ No EP2C papers found to compare")
            return {"total": 0, "results": []}
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPARISON")
        print(f"{'='*70}")
        print(f"Found {len(papers)} EP2C papers to compare\n")
        
        results = []
        for i, paper_info in enumerate(papers, 1):
            print(f"[{i}/{len(papers)}] {paper_info['paper_name']}")
            result = self.compare_paper(
                paper_info=paper_info,
                compare_with_gold=compare_with_gold,
                run_evaluation=run_evaluation
            )
            results.append(result)
        
        # Generate summary statistics
        summary = self._generate_summary(results)
        
        return {
            "total": len(papers),
            "results": results,
            "summary": summary
        }
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics from comparison results."""
        summary = {
            "total_papers": len(results),
            "ep2c_evaluated": 0,
            "gold_evaluated": 0,
            "ep2c_scores": [],
            "gold_scores": [],
            "score_differences": [],
        }
        
        for result in results:
            ep2c_eval = result.get("ep2c_evaluation", {})
            gold_eval = result.get("gold_evaluation", {})
            
            if isinstance(ep2c_eval, dict) and "score" in ep2c_eval:
                summary["ep2c_evaluated"] += 1
                summary["ep2c_scores"].append(ep2c_eval["score"])
            
            if isinstance(gold_eval, dict) and "score" in gold_eval:
                summary["gold_evaluated"] += 1
                summary["gold_scores"].append(gold_eval["score"])
            
            if "score_difference" in result:
                summary["score_differences"].append(result["score_difference"])
        
        # Calculate averages
        if summary["ep2c_scores"]:
            summary["avg_ep2c_score"] = sum(summary["ep2c_scores"]) / len(summary["ep2c_scores"])
        
        if summary["gold_scores"]:
            summary["avg_gold_score"] = sum(summary["gold_scores"]) / len(summary["gold_scores"])
        
        if summary["score_differences"]:
            summary["avg_score_difference"] = sum(summary["score_differences"]) / len(summary["score_differences"])
        
        return summary
    
    def save_comparison(self, comparison: Dict, output_path: Path):
        """Save comparison results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Comparison results saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare EP2C pipeline results with Paper2Code gold repositories",
        epilog="""
Examples:
  # Compare all EP2C papers with gold repositories
  python compare_pipelines.py --ep2c_output_dir ./outputs --gold_repos_dir ./gold_repos

  # Compare without evaluation (just find repos)
  python compare_pipelines.py --ep2c_output_dir ./outputs --no_evaluation

  # Compare specific papers only
  python compare_pipelines.py --ep2c_output_dir ./outputs --papers ACT auto-j
        """
    )
    
    parser.add_argument(
        "--ep2c_output_dir",
        type=Path,
        required=True,
        help="Directory containing EP2C pipeline outputs"
    )
    
    parser.add_argument(
        "--gold_repos_dir",
        type=Path,
        help="Directory containing gold repositories (from Paper2Code dataset)"
    )
    
    parser.add_argument(
        "--eval_gpt_version",
        type=str,
        default="gpt-4o",
        help="GPT version for evaluation (default: gpt-4o)"
    )
    
    parser.add_argument(
        "--no_evaluation",
        action="store_true",
        help="Don't run evaluation, just find and compare repositories"
    )
    
    parser.add_argument(
        "--no_gold",
        action="store_true",
        help="Don't compare with gold repositories, only evaluate EP2C results"
    )
    
    parser.add_argument(
        "--papers",
        nargs="+",
        help="Specific paper names to compare (default: all found papers)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for comparison results (default: comparison_results.json)"
    )
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = PipelineComparator(
        ep2c_output_dir=args.ep2c_output_dir,
        gold_repos_dir=args.gold_repos_dir,
        eval_gpt_version=args.eval_gpt_version
    )
    
    # Run comparison
    comparison = comparator.compare_all(
        compare_with_gold=not args.no_gold,
        run_evaluation=not args.no_evaluation,
        paper_filter=args.papers
    )
    
    # Print summary
    summary = comparison.get("summary", {})
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"Total Papers: {summary.get('total_papers', 0)}")
    
    if not args.no_evaluation:
        print(f"\n📊 Evaluation Results:")
        print(f"   EP2C Evaluated: {summary.get('ep2c_evaluated', 0)}")
        if summary.get('avg_ep2c_score') is not None:
            print(f"   Average EP2C Score: {summary['avg_ep2c_score']:.2f}/5")
        
        if not args.no_gold:
            print(f"   Gold Evaluated: {summary.get('gold_evaluated', 0)}")
            if summary.get('avg_gold_score') is not None:
                print(f"   Average Gold Score: {summary['avg_gold_score']:.2f}/5")
            
            if summary.get('avg_score_difference') is not None:
                print(f"   Average Score Difference: {summary['avg_score_difference']:+.2f}")
    
    print(f"{'='*70}\n")
    
    # Save results
    output_path = args.output or (args.ep2c_output_dir / "comparison_results.json")
    comparator.save_comparison(comparison, output_path)


if __name__ == "__main__":
    main()

