#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path
from eval import Paper2CodeEvaluator

def find_repo_and_paper(subdir):
    repo = None
    paper = None
    
    # Derive the paper_name prefix from the folder name
    # subdir: something like "my_paper"
    paper_prefix = subdir.name

    # Expected exact filenames
    md_name = f"{paper_prefix}.md"
    json_name = f"{paper_prefix}_cleaned.json"

    for p in Path(subdir).iterdir():
        if p.is_dir() and p.name.endswith("_repo"):
            repo = p
        
        # STRICT matching: exact same filename only
        if p.name == md_name:
            paper = p
        elif p.name == json_name:
            # Only accept JSON if no MD was found
            if paper is None:
                paper = p

    return repo, paper

def main():
    parser = argparse.ArgumentParser(description='Batch eval repo folders reference-free with eval.py.')
    parser.add_argument('--eval_root', type=str, required=True, help='Folder containing subfolders with _repo and paper .jsons')
    parser.add_argument('--gpt_version', type=str, default='gpt-4o')
    parser.add_argument('--output', type=str, default='batch_eval_summary.json')
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    evaluator = Paper2CodeEvaluator(gpt_version=args.gpt_version)
    summary = []

    for sub in sorted(eval_root.iterdir()):
        if not sub.is_dir():
            continue
        repo, paper = find_repo_and_paper(sub)
        if not repo or not paper:
            print(f"Skipping {sub.name}: missing repo or paper JSON.")
            continue
        print(f"Evaluating: {sub.name} | repo: {repo.name} | paper: {paper.name}")
        try:
            result = evaluator.evaluate_ref_free(str(paper), str(repo))
            score = result.get('score', None)
            summary.append({'folder': sub.name, 'repo': repo.name, 'paper': paper.name, 'score': score})
            print(f"Score: {score}")
        except Exception as e:
            print(f"Error evaluating {sub.name}: {e}")
            summary.append({'folder': sub.name, 'repo': repo.name, 'paper': paper.name, 'score': None, 'error': str(e)})

    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nBatch evaluation complete. Results saved to {output_path}")
    scored = [s for s in summary if s['score'] is not None]
    if scored:
        mean = sum([float(s['score']) for s in scored])/len(scored)
        print(f"Mean score: {mean:.2f}")

if __name__ == '__main__':
    main()
