# app.py
import os
import yaml
import logging
import time
import numpy as np

from datasets import DatasetLoader
from environment import Environment
from prompt_manager import PromptManager
from pddl_utils import PDDLUtils
from planner_interface import PlannerInterface
from refinement_controller import RefinementController

def main():
    # Set up logger
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Extract global parameters
    n_p = config['generation'].get('n_p', 5)
    n_d = config['generation'].get('n_d', 10)
    c_max = config['generation'].get('c_max', 4)
    T_max = config['generation'].get('T_max', 10)
    temp_init = config['generation'].get('temperature_initial', 0.0)
    temp_refine = config['generation'].get('temperature_refinement', 0.7)
    plans_per_problem = config['evaluation'].get('plans_per_problem', 1)
    success_threshold = config['evaluation'].get('success_threshold', 0.5)
    
    # Load prompt templates
    templates = {
        'initial_prompt_template': config['prompting'].get('initial_prompt_template', ''),
        'refinement_prompt_template': config['prompting'].get('refinement_prompt_template', ''),
        'problem_prompt_template': config['prompting'].get('problem_prompt_template', '')
    }
    prompt_manager = PromptManager(templates)
    
    # Load dataset
    dataset_loader = DatasetLoader(root_path='datasets')  # Update as needed
    env_names = dataset_loader.get_environment_names()
    logger.info(f"Loaded environments: {env_names}")
    
    # Instantiate utilities
    pddl_utils = PDDLUtils()
    # Initialize planner with path from config or default path
    planner_path = config.get('planner_path', 'path/to/fast-downward.py')  # Replace or configure accordingly
    planner = PlannerInterface(planner_path=planner_path)
    
    # For each environment, run refinement
    all_results = []
    for env_name in env_names:
        logger.info(f"Starting environment: {env_name}")
        # Load environment data
        ground_truth_domain, ground_truth_problems = dataset_loader.get_ground_truth_pddl(env_name)
        nl_descs = dataset_loader.get_nl_descriptions(env_name)
        env_objects = dataset_loader.get_object_list(env_name)
        env_action_iface = dataset_loader.get_action_interface(env_name)
        
        # Create environment instance
        env_obj = Environment(env_id=env_name, config={'environment': {'api_base_url': config['environment'].get('api_base_url', '')}})
        env_obj.reset()
        
        # Setup refinement controller
        ref_ctrl = RefinementController(
            environment=env_obj,
            prompt_manager=prompt_manager,
            pddl_utils=pddl_utils,
            planner=planner,
            env_name=env_name,
            ground_truth_domain=ground_truth_domain,
            ground_truth_problems=ground_truth_problems,
            nl_descriptions=nl_descs,
            environment_objects=env_objects
        )
        
        # Run the iterative refinement process
        result = ref_ctrl.execute_refinement(
            nl_descriptions=nl_descs,
            nl_task=nl_descs.get('problems', ''),  # Using problem NL as task description
            max_iterations=c_max
        )
        # Append environment results
        all_results.append({
            'environment': env_name,
            'result': result
        })
        
        # Save the final generated PDDL files to designated directory
        out_dir = os.path.join('results', env_name)
        os.makedirs(out_dir, exist_ok=True)
        pddl_utils.write_pddl_files(result['final_domain'], result['final_problem'], out_dir)
        logger.info(f"Saved final PDDL files for {env_name} in {out_dir}")
        
        # Optional: plan validation and reporting
        # Can be added here if further validation is needed

    # Summarize overall results
    total_envs = len(env_names)
    total_successful = sum(1 for r in all_results if any(r['result']['solve_rates']))
    overall_solve_rate = total_successful / total_envs if total_envs > 0 else 0.0
    logger.info(f"Overall task solve rate across environments: {overall_solve_rate * 100:.2f}%")
    
    # Print detailed report
    for r in all_results:
        env = r['environment']
        res = r['result']
        solve_rate = np.mean(res['solve_rates']) if res['solve_rates'] else 0.0
        ew_avg = np.mean(res['ew_scores']) if res['ew_scores'] else 0.0
        cycles = res['iterations']
        logger.info(f"Environment: {env}")
        logger.info(f"  Success rate: {solve_rate*100:.1f}%")
        logger.info(f"  EW score: {ew_avg:.3f}")
        logger.info(f"  Refinement cycles: {cycles}")
    
    # End of main
    logger.info("Experiment completed.")

if __name__ == '__main__':
    main()
