## main.py
import argparse
import os
import sys
import yaml
import json
import random
import torch
import numpy as np

from utils import load_config, setup_logging
from dataset_loader import DatasetLoader
from model import Model
from profiling import ProfileAnalyzer
from steering import AttentionReweighter
from evaluation import Evaluation
from utils import create_prompt

def parse_args():
    parser = argparse.ArgumentParser(description="Main script for PASTA experiments")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--tasks', nargs='+', default=None, help='List of tasks to run, default: all')
    parser.add_argument('--do_profiling', action='store_true', help='Run attention head profiling')
    parser.add_argument('--do_inference', action='store_true', help='Run inference with attention steering')
    parser.add_argument('--do_evaluation', action='store_true', help='Evaluate generated outputs')
    parser.add_argument('--profile_tasks', nargs='+', default=None, help='Tasks to profile on')
    parser.add_argument('--test_tasks', nargs='+', default=None, help='Tasks to evaluate on')
    parser.add_argument('--load_profile', type=str, default=None, help='Path to precomputed profile heads JSON')
    parser.add_argument('--k_heads', type=int, default=None, help='Number of heads to steer; overrides config if set')
    args = parser.parse_args()
    return args

def main():
    # Parse command-line arguments
    args = parse_args()
    setup_logging()

    # Load config
    cfg = load_config(args.config)

    # Set device
    device = cfg['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, falling back to CPU.")

    # Load datasets
    dataset_paths = cfg.get('datasets', {}).get('dataset_paths', {})
    task_list = args.tasks if args.tasks else ['BiasBios', 'CounterFact', 'JSON Formatting', 'Pronouns Changing']
    # If specific tasks provided in args, override
    task_list = args.tasks if args.tasks else task_list

    # Initialize DatasetLoader
    dataset_loader = DatasetLoader(dataset_paths=dataset_paths, task_name=None)
    datasets_by_task = {}
    for task in task_list:
        dataset_loader.task_name = task
        datasets_by_task[task] = dataset_loader.load_dataset()

    # Initialize model
    model_name = cfg['training'].get('model_name', 'llama-7b')
    model = Model(model_name=model_name, device=device)
    model.eval()

    # Optional: Load attention hooks (done inside model.py during init)
    # For safety, attach hooks now (done inside Model class constructor)

    # Profile attention heads if needed
    profile_results_path = args.load_profile
    selected_heads = []
    if args.do_profiling:
        print("Starting profiling of attention heads...")
        # For profiling, use small dataset (profile_samples, e.g., 1000)
        profile_dataset_list = []
        for task in task_list:
            # Use first 1000 samples for profiling
            profile_dataset_list.extend(datasets_by_task[task]['train'][:cfg['training'].get('profiling_samples', 1000)])
        for task in task_list:
            print(f"Profiling task: {task}")
            profile_fetch = datasets_by_task[task]['train'][:cfg['training'].get('profiling_samples', 1000)]
            profile_analyzer = ProfileAnalyzer(
                model=model,
                profile_dataset=profile_fetch,
                task_name=task,
                config=cfg
            )
            selected_heads_task = profile_analyzer.profile_heads()
            # Save or accumulate profile heads
            # For multi-task, take intersection across all tasks
            if not selected_heads:
                selected_heads = set(selected_heads_task)
            else:
                selected_heads = selected_heads.intersection(selected_heads_task)
        # Convert to list and pick top-K if needed
        selected_heads = list(selected_heads)
        # Save profile heads
        profile_out_path = f'profile_heads_{"+".join(task_list)}.json'
        with open(profile_out_path, 'w') as f:
            json.dump([list(h) for h in selected_heads], f)
        print(f"Profile heads saved to {profile_out_path}")
    elif args.load_profile:
        # Load precomputed profile heads
        with open(args.load_profile, 'r') as f:
            selected_heads = [tuple(h) for h in json.load(f)]
        print(f"Loaded profile heads from {args.load_profile}")
    else:
        # Use heads specified in config or default
        default_k = cfg['training'].get('top_k_heads', 400)
        # If no profile, select top-k randomly or use shared default
        selected_heads = []  # if empty, no steering
        print("No profiling performed; no heads will be steered unless specified.")

    # Initialize attention reweighter
    alpha = cfg['training'].get('alpha', 0.01)
    bump_heads = selected_heads
    attention_weighter = AttentionReweighter(head_indices=bump_heads, alpha=alpha)

    # For each task, prepare prompts, perform inference, evaluate
    results_by_task = {}
    for task in task_list:
        print(f"\n=== Processing task: {task} ===")
        dataset = datasets_by_task[task]['test']  # use test split for evaluation
        prompts_template = cfg.get('prompts', {}).get(f"{task}_template", None)
        if prompts_template is None:
            print(f"No prompt template found for task {task}, skipping.")
            continue

        # For evaluation, prepare list to store outputs
        gen_texts = []
        task_samples = dataset[:]
        # For each sample, generate prompt with emphasis
        for sample in task_samples:
            raw_input_text = sample.get('input_text', '')
            # Assume 'highlighted_spans' are already embedded or known
            # For this problem, generate spans from dataset if in dataset
            # Otherwise, assume dataset has 'highlighted_spans' info
            hl_indices = sample.get('highlighted_spans', [])
            # Generate prompt
            prompt_str = create_prompt(
                template_str=prompts_template,
                input_text=raw_input_text,
                highlighted_spans=hl_indices,
                instruction=sample.get('task_instruction', ''),
                emphasis_marker='**'
            )

            # Tokenize prompt
            encodings = model.tokenizer(prompt_str, return_tensors='pt').to(device)
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']

            # Generate with attention steering
            output_text = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg['training'].get('max_sequence_length', 512),
                do_sample=False
            )

            gen_texts.append(output_text)

        # Evaluation
        if args.do_evaluation:
            evaluation = Evaluation(
                model_outputs=gen_texts,
                dataset=task_samples,
                task_name=task
            )
            metrics_dict = evaluation.evaluate()
            results_by_task[task] = metrics_dict
            print(f"Results for {task}: {metrics_dict}")

    # Final summarize or save results
    results_path = f'results_summary_{"+".join(task_list)}.json'
    with open(results_path, 'w') as f:
        json.dump(results_by_task, f, indent=2)
    print(f"All results saved to {results_path}")

if __name__ == '__main__':
    main()
