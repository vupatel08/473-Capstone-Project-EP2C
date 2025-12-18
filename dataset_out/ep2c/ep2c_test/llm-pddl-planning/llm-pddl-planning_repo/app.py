# app.py
import os
import time
import json
import logging
from typing import List, Tuple, Dict, Optional, Union
import yaml

# Import required modules/classes
from utils import (
    log_results,
    generate_prompt_initial_problem,
    generate_prompt_initial_domain,
    refine_prompt
)
from environment import Environment
from llm_manager import LLManager
from pddl_generator import PDDLGenerator
from metrics import (
    calculate_ew_score,
    evaluate_plan_success
)
from refinement import Refinement

# --- Load configuration ---
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    # Fallback default config
    config = {
        'environment': {
            'max_sampling_length': 10,
            'num_samples': 4
        },
        'refinement': {
            'max_refinement_iterations': 20,
            'ew_threshold': 0.84,
            'plan_success_rate_threshold': 0.66
        },
        'logging': {
            'log_dir': './logs',
            'log_level': 'INFO'
        }
    }

# Set up logging
log_dir = config['logging'].get('log_dir', './logs')
os.makedirs(log_dir, exist_ok=True)
log_level_str = config['logging'].get('log_level', 'INFO')
log_level = getattr(logging, log_level_str.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'app.log')),
        logging.StreamHandler()
    ]
)

# Extract parameters
T_MAX = config['environment'].get('max_sampling_length', 10)
NUM_SAMPLES = config['environment'].get('num_samples', 4)
MAX_REFINEMENT_ITER = config['refinement'].get('max_refinement_iterations', 20)
EW_THRESHOLD = config['refinement'].get('ew_threshold', 0.84)
PLAN_THRESH = config['refinement'].get('plan_success_rate_threshold', 0.66)

# Placeholder environment info (to be replaced with actual data)
# For all environments, define environment descriptions, object lists, API signatures
# For simplicity, this example assumes preloaded info
# In real usage, replace/add environment-specific info here.
env_descriptions = {
    'env1': {
        'nl_desc': 'Description of environment 1',
        'object_list': ['obj1', 'obj2', 'obj3'],
        'api_signatures': {}  # e.g., action signatures
    },
    # Add more environments if needed
}
task_descriptions = {
    'env1': {
        'nl_task': 'NL description of task 1 in env1',
    },
    # Add more tasks if needed
}

# Assume environment signatures are known. For mockup:
default_api_signatures = {}  # populate as per environment

# --- Initialize core components ---
# Environment
# For actual implementation, environment info may need to be environment-specific
# Here, we create a generic environment object
env = Environment(api_signatures=default_api_signatures, env_id='env1')

# LLM manager
api_key = os.getenv('OPENAI_API_KEY', 'YOUR_API_KEY_HERE')  # replace as needed
llm_manager = LLManager(api_key=api_key)

# PDDL generator
pddl_generator = PDDLGenerator()

# Metrics
# No initialization needed
# Refinement
refiner = None  # instantiate per environment/task

# --- Main execution ---
results_summary = []

for env_id, descs in env_descriptions.items():
    nl_env_desc = descs['nl_desc']
    object_list = descs['object_list']
    api_sigs = descs['api_signatures']
    # For each environment, process associated tasks
    for task_id, task_desc in task_descriptions.get(env_id, {}).items():
        nl_task_desc = task_desc['nl_task']

        # Generate initial problem and domain candidates
        logging.info(f"Processing environment '{env_id}', task '{task_id}'")
        # Generate initial problem PDDL
        problem_candidates = llm_manager.generate_initial_problem(nl_task_desc, object_list, api_sigs)
        # Generate initial domain PDDL
        domain_candidate = llm_manager.generate_initial_domain(nl_env_desc, api_sigs)

        # Save initial problem/domain files
        with open('initial_problem.pddl', 'w') as f:
            f.write(problem_candidates)
        with open('initial_domain.pddl', 'w') as f:
            f.write(domain_candidate)
        current_problem_path = 'initial_problem.pddl'
        current_domain_path = 'initial_domain.pddl'

        # Initialize refinement class
        refiner = Refinement(env, llm_manager)

        # Start iterative refinement
        start_time = time.time()
        best_domain_str = domain_candidate
        best_problem_str = problem_candidates
        best_ew_score = 0.0
        plan_solved = False
        plan_success_rate = 0.0
        final_iterations = 0

        for iteration in range(1, MAX_REFINEMENT_ITER + 1):
            logging.info(f"Iteration {iteration}")
            # Attempt to plan using current domain/problem
            plan_str = refiner.solve_in_environment(current_domain_path, current_problem_path)

            if plan_str:
                # Validate in environment
                plan_success = evaluate_plan_success(plan_str, env)
                if plan_success:
                    # Successfully reached goal
                    plan_solved = True
                    logging.info(f"Task '{task_id}' succeeded at iteration {iteration}")
                    break
                else:
                    logging.info(f"Plan invalid or unsuccessful in environment at iteration {iteration}")
            else:
                logging.info(f"No plan found at iteration {iteration}")

            # If plan failed or no plan, compute EW between current model and environment
            try:
                ew_score = calculate_ew_score(
                    current_domain_path,
                    env,
                    T_MAX
                )
            except Exception as e:
                logging.warning(f"EW calculation failed: {e}")
                ew_score = 0.0

            # Generate environment state snapshot for prompting
            env_state = env.get_state()

            # Prepare feedback info
            feedback_info = {
                'ew_score': ew_score,
                'env_state': env_state
            }

            # Generate refined domain PDDL using LLM
            refined_domain_str = llm_manager.refine_prompt(
                current_domain_str,
                ew_score,
                additional_context={'env_state': env_state}
            )

            # Validate syntax
            valid_syntax, _ = pddl_generator.validate_pddl_syntax(refined_domain_str)
            if not valid_syntax:
                logging.warning(f"Refined domain syntax invalid at iteration {iteration}")
                # Could implement retry or skip; here, proceed
                refined_domain_str = current_domain_str

            # Save refined domain temporarily
            with open('refined_domain.pddl', 'w') as f:
                f.write(refined_domain_str)

            # Update current domain for next iteration
            current_domain_path = 'refined_domain.pddl'
            current_domain_str = refined_domain_str

            # Generate plan in new domain
            plan_str = refiner.solve_in_environment(current_domain_path, current_problem_path)

            # Evaluate plan
            if plan_str:
                plan_success = evaluate_plan_success(plan_str, env)
                if plan_success:
                    plan_solved = True
                    logging.info(f"Success after refinement at iteration {iteration}")
                    break

        # Save final logs
        total_time = time.time() - start_time
        resource_usage = {
            'time_seconds': total_time,
            'iterations': iteration
        }
        log_results(
            env_id=env_id,
            success=plan_solved,
            ew_score=ew_score if 'ew_score' in locals() else 0.0,
            iterations=iteration,
            resource_usage=resource_usage
        )
        results_summary.append({
            'env_id': env_id,
            'task_id': task_id,
            'success': plan_solved,
            'iterations': iteration,
            'ew_score': ew_score if 'ew_score' in locals() else None,
            'final_domain': current_domain_str if 'current_domain_str' in locals() else None,
            'final_problem': current_problem_str if 'current_problem_str' in locals() else None,
            'time_seconds': total_time
        })

# Save overall summary
with open(os.path.join(log_dir, 'results_summary.json'), 'w') as f:
    json.dump(results_summary, f, indent=2)

logging.info("All environments processed. Results saved.")
