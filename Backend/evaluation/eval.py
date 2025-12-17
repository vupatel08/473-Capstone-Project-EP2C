#!/usr/bin/env python3
"""
Paper2Code Evaluation Script
Evaluates generated code repositories against research papers using LLM-based evaluation.

Supports two evaluation modes:
1. Reference-based: Compares target repository against a gold repository
2. Reference-free: Evaluates target repository directly against the paper
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import re

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

from openai import OpenAI
from openai import RateLimitError
import time


class Paper2CodeEvaluator:
    """Evaluates generated code repositories against research papers."""
    
    def __init__(self, gpt_version: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            gpt_version: GPT model version to use for evaluation
            api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
        """
        self.gpt_version = gpt_version
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Set it in environment or pass as argument.")
        self.client = OpenAI(api_key=api_key)
        
        # Load evaluation prompts
        prompts_dir = backend_dir / "prompts"
        self.ref_based_prompt = self._load_prompt(prompts_dir / "ref_based.txt")
        self.ref_free_prompt = self._load_prompt(prompts_dir / "ref_free.txt")
    
    def _load_prompt(self, prompt_path: Path) -> str:
        """Load evaluation prompt from file."""
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_paper_content(self, paper_path: str) -> str:
        """
        Load paper content from JSON or markdown file.
        Args:
            paper_path: Path to paper JSON or markdown file
        Returns:
            Paper content as string
        """
        paper_path = Path(paper_path)
        if not paper_path.exists():
            raise FileNotFoundError(f"Paper file not found: {paper_path}")
        
        if paper_path.suffix.lower() == '.json':
            with open(paper_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)

            # Special handling for S2ORC/Paper2Code '_cleaned.json' and similar
            if paper_path.name.endswith('_cleaned.json'):
                content_parts = []
                if 'title' in paper_data:
                    content_parts.append(f"Title: {paper_data['title']}")
                if 'abstract' in paper_data:
                    content_parts.append(f"Abstract: {paper_data['abstract']}")
                # S2ORC sometimes puts text in 'sections', 'body_text', or 'paragraphs'. Try all.
                body_candidates = []
                if 'sections' in paper_data:
                    # S2ORC: Each section is typically a dict with 'text' and 'section_title'
                    for section in paper_data['sections']:
                        title = section.get('section_title','') if isinstance(section, dict) else ''
                        text = section.get('text','') if isinstance(section, dict) else ''
                        if title:
                            content_parts.append(f"## {title}")
                        if text:
                            content_parts.append(text)
                elif 'body_text' in paper_data:
                    # Sometimes list of dicts with 'text'
                    for item in paper_data['body_text']:
                        if isinstance(item, dict) and 'text' in item:
                            content_parts.append(item['text'])
                        elif isinstance(item, str):
                            content_parts.append(item)
                elif 'paragraphs' in paper_data:
                    for para in paper_data['paragraphs']:
                        content_parts.append(para)
                elif 'text' in paper_data:  # fallback
                    content_parts.append(paper_data['text'])
                return '\n\n'.join(content_parts)
            # The legacy path for paper_content.json/parsed format
            else:
                content_parts = []
                if 'title' in paper_data:
                    content_parts.append(f"Title: {paper_data['title']}")
                if 'abstract' in paper_data:
                    content_parts.append(f"Abstract: {paper_data['abstract']}")
                if 'body_text' in paper_data:
                    for section in paper_data['body_text']:
                        if isinstance(section, dict) and 'text' in section:
                            content_parts.append(section['text'])
                        elif isinstance(section, str):
                            content_parts.append(section)
                return "\n\n".join(content_parts)
        
        if paper_path.suffix.lower() == ".md":
            with open(paper_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _load_repository_code(self, repo_path: str, max_files: int = 50, max_tokens: int = 100000) -> str:
        """
        Load code from a repository directory.
        
        Args:
            repo_path: Path to repository directory
            max_files: Maximum number of files to include
            max_tokens: Maximum tokens to include (approximate)
            
        Returns:
            Repository code as formatted string
        """
        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")
        
        code_files = []
        total_size = 0
        
        # Collect Python files
        for py_file in sorted(repo_path.rglob("*.py")):
            if total_size > max_tokens:
                break
            if len(code_files) >= max_files:
                break
            
            # Skip common non-code directories
            skip_dirs = {'__pycache__', '.git', 'venv', 'env', 'node_modules', '.pytest_cache'}
            if any(skip_dir in py_file.parts for skip_dir in skip_dirs):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    rel_path = py_file.relative_to(repo_path)
                    code_files.append({
                        'path': str(rel_path),
                        'content': content
                    })
                    total_size += len(content.split())  # Approximate token count
            except Exception as e:
                print(f"Warning: Could not read {py_file}: {e}", file=sys.stderr)
                continue
        
        # Format as string
        formatted_code = []
        for file_info in code_files:
            formatted_code.append(f"File: {file_info['path']}")
            formatted_code.append("```python")
            formatted_code.append(file_info['content'])
            formatted_code.append("```")
            formatted_code.append("")
        
        return "\n".join(formatted_code)
    
    def _call_llm(self, messages: List[Dict[str, str]], max_retries: int = 5, base_delay: float = 1.0) -> str:
        """
        Call LLM with retry logic for rate limits.
        
        Args:
            messages: List of message dicts for the API
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            
        Returns:
            LLM response content
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_version,
                    messages=messages,
                    temperature=0.0
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                
                # Try to extract retry-after time from error message
                error_message = str(e)
                retry_after = None
                
                match = re.search(r'try again in ([\d.]+)s', error_message, re.IGNORECASE)
                if match:
                    retry_after = float(match.group(1))
                
                if retry_after is None:
                    retry_after = base_delay * (2 ** attempt)
                
                jitter = retry_after * 0.1 * (0.5 + (hash(str(messages)) % 100) / 100)
                wait_time = retry_after + jitter
                
                print(f"⚠️  Rate limit reached (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.2f}s...", flush=True)
                time.sleep(wait_time)
            except Exception as e:
                raise
    
    def _parse_evaluation_response(self, response: str) -> Dict:
        """
        Parse evaluation response from LLM.

        Args:
            response: LLM response string

        Returns:
            Parsed evaluation result dictionary
        """
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                # Ensure score is rounded properly if present
                if "score" in result:
                    try:
                        result["score"] = round(float(result["score"]), 2)
                    except:
                        result["score"] = None
                return result
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract score (allow decimals)
        score_match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', response)
        if score_match:
            score = round(float(score_match.group(1)), 2)
        else:
            score = None

        return {
            "score": score,
            "raw_response": response,
            "critique_list": []
    }
    
    def evaluate_ref_based(
        self,
        paper_path: str,
        target_repo_path: str,
        gold_repo_path: str
    ) -> Dict:
        """
        Evaluate target repository against gold repository (reference-based).
        
        Args:
            paper_path: Path to paper JSON or markdown file
            target_repo_path: Path to target (generated) repository
            gold_repo_path: Path to gold (reference) repository
            
        Returns:
            Evaluation result dictionary with score and critiques
        """
        print("Loading paper content...", flush=True)
        paper_content = self._load_paper_content(paper_path)
        
        print("Loading target repository code...", flush=True)
        target_code = self._load_repository_code(target_repo_path)
        
        print("Loading gold repository code...", flush=True)
        gold_code = self._load_repository_code(gold_repo_path)
        
        # Format prompt
        prompt = self.ref_based_prompt.replace("{{Paper}}", paper_content)
        prompt = prompt.replace("{{Code}}", target_code)
        prompt = prompt.replace("{{GoldCode}}", gold_code)
        
        print(f"Calling {self.gpt_version} for evaluation...", flush=True)
        messages = [{"role": "user", "content": prompt}]
        response = self._call_llm(messages)
        
        result = self._parse_evaluation_response(response)
        result["evaluation_type"] = "ref_based"
        result["paper_path"] = paper_path
        result["target_repo_path"] = target_repo_path
        result["gold_repo_path"] = gold_repo_path
        
        return result
    
    def evaluate_ref_free(
        self,
        paper_path: str,
        target_repo_path: str
    ) -> Dict:
        """
        Evaluate target repository directly against paper (reference-free).
        
        Args:
            paper_path: Path to paper JSON or markdown file
            target_repo_path: Path to target (generated) repository
            
        Returns:
            Evaluation result dictionary with score and critiques
        """
        print("Loading paper content...", flush=True)
        paper_content = self._load_paper_content(paper_path)
        
        print("Loading target repository code...", flush=True)
        target_code = self._load_repository_code(target_repo_path)
        
        # Format prompt
        prompt = self.ref_free_prompt.replace("{{Paper}}", paper_content)
        prompt = prompt.replace("{{Code}}", target_code)
        
        print(f"Calling {self.gpt_version} for evaluation...", flush=True)
        messages = [{"role": "user", "content": prompt}]
        response = self._call_llm(messages)
        
        result = self._parse_evaluation_response(response)
        result["evaluation_type"] = "ref_free"
        result["paper_path"] = paper_path
        result["target_repo_path"] = target_repo_path
        
        return result


def evaluate_from_pipeline(
    paper_path: str,
    target_repo_path: str,
    gold_repo_path: Optional[str] = None,
    gpt_version: str = "gpt-4o",
    output_path: Optional[str] = None
) -> Dict:
    """
    Evaluate a generated repository from the unified pipeline.
    
    This is a convenience function for use with the unified pipeline.
    
    Args:
        paper_path: Path to paper JSON or markdown file
        target_repo_path: Path to target (generated) repository
        gold_repo_path: Optional path to gold (reference) repository
        gpt_version: GPT model version to use
        output_path: Optional path to save evaluation results
        
    Returns:
        Evaluation result dictionary
    """
    evaluator = Paper2CodeEvaluator(gpt_version=gpt_version)
    
    if gold_repo_path:
        result = evaluator.evaluate_ref_based(
            paper_path=paper_path,
            target_repo_path=target_repo_path,
            gold_repo_path=gold_repo_path
        )
    else:
        result = evaluator.evaluate_ref_free(
            paper_path=paper_path,
            target_repo_path=target_repo_path
        )
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated code repositories against research papers",
        epilog="Example: python eval.py --paper paper.json --target_repo generated_repo --gold_repo gold_repo"
    )
    
    parser.add_argument("--paper", type=str, required=True, help="Path to paper JSON or markdown file")
    parser.add_argument("--target_repo", type=str, required=True, help="Path to target (generated) repository")
    parser.add_argument("--gold_repo", type=str, default=None, help="Path to gold (reference) repository (for ref-based eval)")
    parser.add_argument("--gpt_version", type=str, default="gpt-4o", help="GPT model version (default: gpt-4o)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path (default: print to stdout)")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Paper2CodeEvaluator(gpt_version=args.gpt_version)
    
    # Run evaluation
    if args.gold_repo:
        print("="*70)
        print("REFERENCE-BASED EVALUATION")
        print("="*70)
        result = evaluator.evaluate_ref_based(
            paper_path=args.paper,
            target_repo_path=args.target_repo,
            gold_repo_path=args.gold_repo
        )
    else:
        print("="*70)
        print("REFERENCE-FREE EVALUATION")
        print("="*70)
        result = evaluator.evaluate_ref_free(
            paper_path=args.paper,
            target_repo_path=args.target_repo
        )
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Evaluation results saved to: {args.output}")
    else:
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Evaluation Type: {result.get('evaluation_type', 'unknown')}")
    print(f"Score: {result.get('score', 'N/A')}")
    print(f"Number of Critiques: {len(result.get('critique_list', []))}")
    print("="*70)


if __name__ == "__main__":
    main()

