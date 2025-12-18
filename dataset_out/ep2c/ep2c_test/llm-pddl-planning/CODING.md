# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## app.py

```python
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
```

**Notes:**
- Replace placeholder environment info, prompt templates, API keys, and environment API calls as needed.
- This code follows the architecture, logic flow, and data structures outlined in the plan, ensuring modularity and correctness.
- The `refine_prompt` and plan solving functions rely on existing modules and proper environment setup.
- Ensure the environment API, plan parser, and plan executor are correctly integrated for real use.

## environment.py

```python
## environment.py

import re
import copy
import random
from typing import List, Dict, Tuple, Union, Optional

# For resource measurement, environment specifics, and simulation
import logging

# We assume a simplified environment interface.
# In a real setting, replace or extend this with actual environment SDK/API.
# Here, for illustration, we simulate a generic environment with minimal logic.

class Environment:
    def __init__(self, api_signatures: Dict[str, Dict], env_id: str = "default_env"):
        """
        Initialize environment with given API signatures and environment ID.
        api_signatures: dict mapping action names to parameter info.
        env_id: environment identifier, for logging or environment-specific logic.
        """
        self.api_signatures = api_signatures
        self.env_id = env_id
        # Internal state: a dictionary to hold environment-specific info
        # For illustration, we initialize with default state
        self.state = self._initialize_state()
        # Store environment-specific info if needed
        # For custom environments, override or extend as needed.
        # For example, object positions, status flags, etc.
        # For generality, use a dict.
        # Deep copy for reset capabilities
        self._initial_state = copy.deepcopy(self.state)
        # Setup environment parameters for simulation
        self.actions_list = list(self.api_signatures.keys())

    def _initialize_state(self) -> Dict:
        """
        Initialize environment state.
        For real environment, replace with actual initialization.
        """
        # For illustration, a generic environment with object locations
        # e.g., object positions, robot positions, flags
        # Here, we create a dummy state - extend as needed per environment
        state = {
            'objects': {},  # e.g., {'object1': location, ...}
            'robot_positions': {},  # e.g., {'robot1': 'room2'}
            'object_status': {},  # e.g., {'object1': 'carried'/'free', ...}
            'flags': {}  # other environment-specific flags
        }
        # For demonstration, we leave it empty; in practice, load initial env state.
        return state

    def reset(self):
        """
        Reset environment to initial state.
        """
        self.state = copy.deepcopy(self._initial_state)

    def get_state(self) -> Dict:
        """
        Return a dict describing the current environment state.
        """
        return copy.deepcopy(self.state)

    def execute_plan(self, plan: str) -> bool:
        """
        Execute a sequence of actions in the environment.
        plan: string with actions in PDDL syntax, e.g.,
        (move robot1 room2 room3)
        Returns True if plan executed successfully to goal.
        """
        try:
            actions = self._parse_plan(plan)
            for action_str in actions:
                success = self._execute_action(action_str)
                if not success:
                    # Infeasible action, stop execution
                    return False
            # After executing all actions, could check goal if needed
            # For now, assume success if no errors
            return True
        except Exception as e:
            logging.error(f"Error executing plan in env {self.env_id}: {e}")
            return False

    def _parse_plan(self, plan_str: str) -> List[str]:
        """
        Parse plan string into list of action strings.
        Assumes each action is on a separate line or separated by whitespace.
        """
        # Remove parentheses and split
        actions = re.findall(r'\([^)]*\)', plan_str)
        actions = [action.strip() for action in actions]
        return actions

    def _execute_action(self, action_str: str) -> bool:
        """
        Execute a single action in the environment.
        Return True if feasible and successful; False otherwise.
        """
        # Parse action string, e.g., (move robot1 room2 room3)
        parsed = self._parse_action_str(action_str)
        if not parsed:
            return False
        action_name, params = parsed
        # Verify if action is recognized
        if action_name not in self.api_signatures:
            logging.warning(f"Unrecognized action {action_name} in environment {self.env_id}")
            return False
        # Check preconditions based on current state
        preconditions_ok = self._check_preconditions(action_name, params)
        if not preconditions_ok:
            return False
        # Apply effects
        self._apply_effects(action_name, params)
        # Additional environment-specific checks can be added here
        return True

    def _parse_action_str(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        """
        Parse an action string of the form (action_name param1 param2 ...).
        """
        # Remove parentheses
        if not action_str.startswith('(') or not action_str.endswith(')'):
            return None
        inside = action_str[1:-1].strip()
        parts = inside.split()
        if not parts:
            return None
        action_name = parts[0]
        params = parts[1:]
        return action_name, params

    def _check_preconditions(self, action_name: str, params: List[str]) -> bool:
        """
        Check if preconditions for action are met in current state.
        This is environment specific.
        For illustration, implement dummy checks or extend with environment logic.
        """
        # Here, insert environment logic:
        # For example, if action is 'move', verify robot position, neighbor relations, etc.
        # For now, assume all preconditions are satisfied.
        # Replace with real logic as needed.
        return True

    def _apply_effects(self, action_name: str, params: List[str]) -> None:
        """
        Update the environment state according to effect of action.
        This is environment-specific.
        """
        # For illustration, implement dummy effect updates.
        # Example: for 'move' action, update robot position
        if action_name == 'move' and len(params)==3:
            robot, from_loc, to_loc = params
            self.state['robot_positions'][robot] = to_loc
        elif action_name == 'pick' and len(params)==4:
            robot, obj, room, g = params
            # Mark object as carried
            self.state['object_status'][obj] = 'carried'
            self.state['objects'][obj] = 'carried_by_' + robot
        elif action_name == 'drop' and len(params)==4:
            robot, obj, room, g = params
            # Mark object as at room
            self.state['object_status'][obj] = 'free'
            self.state['objects'][obj] = room
        elif action_name == 'place' and len(params)==4:
            robot, obj, room, g = params
            self.state['object_status'][obj] = 'at_' + room
            self.state['objects'][obj] = room
        elif action_name == 'remove' and len(params)==4:
            robot, obj, room, g = params
            self.state['object_status'][obj] = 'removed'
        elif action_name == 'create' and len(params)==2:
            # Simplify, create new object
            obj, loc = params
            self.state['object_status'][obj] = 'at_' + loc
            self.state['objects'][obj] = loc
        elif action_name == 'destroy' and len(params)==2:
            obj, loc = params
            self.state['object_status'][obj] = 'destroyed'
        # Extend with other actions as defined
        else:
            # Default: do nothing
            pass

    def check_feasibility(self, actions: List[str]) -> List[bool]:
        """
        Verify feasibility of each action in provided sequence.
        Does not modify environment state.
        Returns list of booleans indicating feasibility.
        """
        # Save current state
        saved_state = copy.deepcopy(self.state)
        feasibilities = []
        for action_str in actions:
            feasible = self._execute_action(action_str)
            feasibilities.append(feasible)
            if not feasible:
                # Rollback environment state
                self.state = copy.deepcopy(saved_state)
                break
        # Restore environment to original state after check
        self.state = copy.deepcopy(saved_state)
        return feasibilities

    # Additional methods for sampling sequences
    def sample_action_sequence(self, sequence_length: int) -> List[str]:
        """
        Sample a random executable sequence of actions of given length.
        Uses uniform sampling over possible actions at each step.
        """
        seq = []
        # Save initial state
        temp_state = copy.deepcopy(self.state)
        for _ in range(sequence_length):
            # Determine feasible actions in current state
            feasible_actions = self._get_feasible_actions()
            if not feasible_actions:
                # No feasible actions, stop sampling
                break
            # Select one at random
            action_str = random.choice(feasible_actions)
            # Simulate execution to update state
            feasible = self._execute_action(action_str)
            if feasible:
                seq.append(action_str)
            else:
                # Should not happen as feasible_actions are filtered
                # but check anyway
                break
        # Restore original state
        self.state = copy.deepcopy(temp_state)
        return seq

    def _get_feasible_actions(self) -> List[str]:
        """
        Return a list of all feasible actions in current state.
        For simplicity, generate all actions and filter.
        In a real environment, this could be optimized.
        """
        feasible_actions = []
        for action_name in self.actions_list:
            # Generate all parameter combinations
            param_combinations = self._generate_action_params(action_name)
            for params in param_combinations:
                action_str = self._build_action_str(action_name, params)
                if self._check_preconditions(action_name, params):
                    feasible_actions.append(action_str)
        return feasible_actions

    def _generate_action_params(self, action_name: str) -> List[List[str]]:
        """
        Generate all permissible parameters for an action based on environment.
        In practice, this depends on environment's objects and predicates.
        For illustration, generate sample parameters.
        """
        # Placeholder: in practice, create parameter combos based on object lists and signatures
        # For demonstration, return an empty list to avoid invalid sampling
        # Extend with environment-specific logic
        return []

    def _build_action_str(self, action_name: str, params: List[str]) -> str:
        """
        Build action string in PDDL syntax.
        """
        return '(' + ' '.join([action_name] + params) + ')'

    # Additional helper: set environment state from outside (for reset or test)
    def set_state(self, state: Dict):
        self.state = copy.deepcopy(state)

```

## llm_manager.py

```python
## llm_manager.py

import openai
import os
import logging
from typing import Tuple
import time

class LLManager:
    def __init__(self, api_key: str):
        """
        Initialize the LLManager with API key and load configurations.
        """
        self.api_key = api_key
        openai.api_key = self.api_key

        # Load prompt templates from the configuration
        # Use default templates if not found in 'config.yaml' or environment
        from utils import CONFIG
        self.initial_problem_prompt_template: str = CONFIG['prompting'].get('initial_prompt_template', "")
        self.domain_prompt_template: str = CONFIG['prompting'].get('domain_prompt_template', "")
        self.refinement_prompt_template: str = CONFIG['prompting'].get('refinement_prompt_template', "")

        # Optional: Set default model
        self.model_name: str = "gpt-4"
        # Optional: Set default max tokens for responses
        self.max_tokens: int = 3000

        # Set openai parameters
        self.temperature_initial: float = 0.0  # deterministic for initial problem/domain
        self.temperature_refine: float = 0.7  # allow diversity in refinement


    def _call_openai(self, prompt: str, temperature: float = 0.0, max_tokens: int = 3000) -> str:
        """
        Call OpenAI API with retries and basic error handling.
        """
        retries = 3
        backoff = 2
        for attempt in range(retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                reply = response.choices[0].message['content']
                return reply
            except openai.error.OpenAIError as e:
                logging.warning(f"OpenAI API error on attempt {attempt+1}: {e}")
                time.sleep(backoff ** attempt)
            except Exception as e:
                logging.error(f"Unexpected error during OpenAI call: {e}")
                time.sleep(backoff ** attempt)
        # If all retries fail
        raise RuntimeError("Failed to get response from OpenAI API after multiple attempts.")

    def generate_initial_problem(self, nl_description: str, object_list: list, api_signatures: dict) -> str:
        """
        Generate initial PDDL problem file from NL description, object list, API signatures.
        Returns the PDDL string.
        """
        prompt = self.initial_problem_prompt_template.format(
            nl_description=nl_description,
            object_list="\n".join([f"{obj}" for obj in object_list]),
            api_signatures=json.dumps(api_signatures, indent=2)
        )
        logging.info("Generating initial problem PDDL...")
        response = self._call_openai(prompt, temperature=self.temperature_initial, max_tokens=3000)
        # Response is expected to contain full PDDL problem code
        return response

    def generate_initial_domain(self, nl_description: str, api_signatures: dict) -> str:
        """
        Generate initial domain PDDL from NL description and API signatures.
        Returns the PDDL string.
        """
        prompt = self.domain_prompt_template.format(
            nl_description=nl_description,
            api_signatures=json.dumps(api_signatures, indent=2)
        )
        logging.info("Generating initial domain PDDL...")
        response = self._call_openai(prompt, temperature=self.temperature_initial, max_tokens=3000)
        return response

    def refine_prompt(self, current_domain: str, environment_feedback: Union[str, float], additional_context: dict = None) -> str:
        """
        Generate a refined domain PDDL incorporating environment feedback or EW signals.
        """
        # Prepare context string if additional info provided
        context_str = ""
        if additional_context:
            context_str = "\n".join([f"{k}: {v}" for k, v in additional_context.items()])

        prompt = self.refinement_prompt_template.format(
            current_domain=current_domain,
            environment_feedback=str(environment_feedback),
            additional_context=context_str
        )
        logging.info("Refining domain PDDL based on feedback...")
        response = self._call_openai(prompt, temperature=self.temperature_refine, max_tokens=3000)
        return response
```

## metrics.py

```python
## metrics.py
"""
This module provides functions to evaluate the similarity between domain models using the Exploration Walk (EW) metrics,
to validate plan success, and to quantify domain-model similarity for guiding the refinement process.
It relies on environment interactions such as sampling action sequences, checking executability, and executing plans.
"""

import math
import logging
from typing import List, Tuple, Union
from utils import calculate_ew_score, evaluate_plan_success, compute_similarity

try:
    from environment import Environment
except ImportError:
    # Environment import may depend on project structure.
    # If unavailable, assign a dummy class or raise later.
    Environment = None

# Configuration parameters from config.yaml
import yaml
import os

try:
    with open('config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    # Default parameters
    CONFIG = {}

# Extract max_sampling_length, num_samples
MAX_SAMPLING_LENGTH: int = CONFIG.get('environment', {}).get('max_sampling_length', 10)
NUM_SAMPLES: int = CONFIG.get('environment', {}).get('num_samples', 4)
# Number of sequences to sample for EW calculation
T_MAX: int = MAX_SAMPLING_LENGTH

def calculate_ew_score(
    domain1: str,
    domain2: str,
    environment: 'Environment',
    max_length: int = T_MAX
) -> float:
    """
    Computes the symmetric Exploration Walk (EW) score between two domain models.
    The score reflects how well sequences sampled from domain1 are executable in domain2,
    and vice versa, averaged harmonically.
    """
    try:
        # Sample sequences from domain1
        sequences1 = environment.sample_action_sequences(domain1, max_length, NUM_SAMPLES)
        # Sample sequences from domain2
        sequences2 = environment.sample_action_sequences(domain2, max_length, NUM_SAMPLES)
    except Exception as e:
        logging.error(f"Error sampling sequences for EW: {e}")
        return 0.0

    def measure_similarity(domain_a: str, domain_b: str, seqs_a: List[List[str]], seqs_b: List[List[str]]) -> float:
        """
        Measure the fraction of sequences from domain_a that are executable in domain_b.
        """
        valid_count = 0
        total = len(seqs_a)
        for seq in seqs_a:
            try:
                feasible_in_b = environment.check_feasibility(seq, domain_b)
                if feasible_in_b:
                    valid_count += 1
            except Exception as e:
                logging.warning(f"Error during feasibility check: {e}")
                # Count as not feasible if exception occurs
                pass
        return valid_count / total if total > 0 else 0.0

    score_d_dhat = measure_similarity(domain1, domain2, sequences1, sequences2)
    score_dhat_d = measure_similarity(domain2, domain1, sequences2, sequences1)

    # Harmonic mean for symmetry
    if score_d_dhat + score_dhat_d == 0:
        ew_score = 0.0
    else:
        ew_score = 2.0 * score_d_dhat * score_dhat_d / (score_d_dhat + score_dhat_d)

    return ew_score

def evaluate_plan_success(plan: str, environment: 'Environment') -> bool:
    """
    Checks whether the given plan successfully achieves the goal in the environment.
    Returns True if the plan is executable and goal is achieved.
    """
    try:
        success = environment.execute_plan(plan)
        return success
    except Exception as e:
        logging.warning(f"Error during plan execution or validation: {e}")
        return False

def compute_similarity(
    domain1: str,
    domain2: str,
    environment: 'Environment',
    sample_sequences: List[List[str]] = None,
    max_length: int = T_MAX
) -> float:
    """
    Computes a similarity score between two domain models.
    The method uses sequence sampling and feasibility checks to estimate similarity.
    """
    try:
        if sample_sequences is None:
            sample_sequences = environment.sample_action_sequences(domain1, max_length, NUM_SAMPLES)
    except Exception as e:
        logging.error(f"Error sampling sequences for similarity: {e}")
        return 0.0

    def sequence_overlap(domain_a: str, domain_b: str, sequences: List[List[str]]) -> float:
        """
        Estimate how many sequences are executable in the other domain.
        """
        count_executable = 0
        total = len(sequences)
        for seq in sequences:
            try:
                feasible = environment.check_feasibility(seq, domain_b)
                if feasible:
                    count_executable += 1
            except Exception as e:
                logging.warning(f"Error during similarity check: {e}")
                continue
        return count_executable / total if total > 0 else 0.0

    similarity_a_b = sequence_overlap(domain1, domain2, sample_sequences)
    similarity_b_a = sequence_overlap(domain2, domain1, sample_sequences)

    # Use harmonic mean to combine similarities, ensure values in [0,1]
    if similarity_a_b + similarity_b_a == 0:
        return 0.0
    similarity_score = 2 * similarity_a_b * similarity_b_a / (similarity_a_b + similarity_b_a)
    return similarity_score

```

## pddl_generator.py

```python
## pddl_generator.py

import json
import re
import logging
from typing import Tuple, Optional
from utils import generate_prompt_initial_problem, generate_prompt_initial_domain, refine_prompt
from utils import validate_pddl_syntax

# Load configuration for prompt templates and params
import yaml
try:
    with open('config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    # Fallback default, should not happen if config.yaml provided
    CONFIG = {}

# Default parameters from config
PLAN_MAX_TOKENS: int = 3000
DEFAULT_TEMPERATURE_INITIAL: float = 0.0
DEFAULT_TEMPERATURE_REFINE: float = 0.7

class PDDLGenerator:
    def __init__(self):
        # No instance variables needed; static methods used
        pass

    @staticmethod
    def generate_initial_problem(
        nl_description: str,
        object_list: list,
        api_signatures: dict
    ) -> str:
        """
        Generate initial complete PDDL problem based on NL description, object list, and action signatures.
        """
        # Create prompt
        prompt = generate_prompt_initial_problem(nl_description, object_list, api_signatures)

        # Call LLM
        raw_response = _call_llm(prompt, temperature=DEFAULT_TEMPERATURE_INITIAL, max_tokens=PLAN_MAX_TOKENS)

        # Validate syntax
        valid, pddl_str = validate_pddl_syntax(raw_response)
        if not valid:
            # Could implement retries or corrections; for now, assume first output
            logging.warning("Generated problem PDDL failed syntax validation.")
        return pddl_str

    @staticmethod
    def generate_initial_domain(
        nl_description: str,
        api_signatures: dict
    ) -> str:
        """
        Generate initial complete PDDL domain based on NL description and API signatures.
        """
        prompt = generate_prompt_initial_domain(nl_description, api_signatures)

        raw_response = _call_llm(prompt, temperature=DEFAULT_TEMPERATURE_INITIAL, max_tokens=PLAN_MAX_TOKENS)

        valid, pddl_str = validate_pddl_syntax(raw_response)
        if not valid:
            # Could do retries or minor fixes; trusting first for now
            logging.warning("Generated domain PDDL failed syntax validation.")
        return pddl_str

    @staticmethod
    def refine_domain(
        current_domain: str,
        environment,
        ew_score: float
    ) -> str:
        """
        Generate a refined domain PDDL based on current domain, environment, and EW feedback.
        """
        # For feedback incorporation, we can include EW score
        # Environment feedback might also include errors, logs, or signals
        environment_feedback = ew_score  # Use EW as a signal; can extend to more detailed feedback

        # Optionally, include more environment signals or errors if available
        # For simplicity, only include EW score here
        prompt = refine_prompt(current_domain, environment_feedback)

        raw_response = _call_llm(prompt, temperature=DEFAULT_TEMPERATURE_REFINE, max_tokens=PLAN_MAX_TOKENS)

        valid, pddl_str = validate_pddl_syntax(raw_response)
        if not valid:
            # Could implement retry or corrections; for now, accept raw
            logging.warning("Refined domain PDDL failed syntax validation.")
        return pddl_str

# Internal helper for LLM API call
def _call_llm(prompt: str, temperature: float, max_tokens: int) -> str:
    import openai
    try:
        response = openai.ChatCompletion.create(
            model='gpt-4',  # As per configuration
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0
        )
        reply = response.choices[0].message['content']
        return reply
    except Exception as e:
        logging.error(f"Error during LLM call: {e}")
        return ""

# Optional: syntax validation utility
def validate_pddl_syntax(pddl_str: str) -> Tuple[bool, str]:
    """
    Validate the syntax of the generated PDDL string.
    Return (True, cleaned_string) if syntactically valid.
    Return (False, original_string) if invalid.
    """
    import pddlpy
    try:
        # simple syntax check: try to parse or load
        # For the purposes of validation, attempting to parse
        # Here, use a dummy parse as placeholder
        # In practice, replace with actual PDDL syntax validation
        if not pddl_str.strip().startswith('(define'):
            return False, pddl_str
        # Additional validation can be added with parser
        return True, pddl_str
    except Exception:
        return False, pddl_str

```

## refinement.py

```python
## refinement.py
"""
This module manages the iterative refinement of the PDDL domain (and optionally problem),
guided by environment interaction, EW metrics, and LLM-based domain generation.
It interfaces with the environment for sampling, plan validation, and execution feedback,
and with the LLManager for generating refined domain models.
"""

import time
import logging
from typing import Tuple, Optional

# Import configuration parameters
import yaml

try:
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    # Default config fallback if file missing, should not happen
    CONFIG = {}

# Extract thresholds and limits from config
MAX_REFINE_ITER: int = CONFIG.get("refinement", {}).get("max_refinement_iterations", 20)
EW_THRESHOLD: float = CONFIG.get("refinement", {}).get("ew_threshold", 0.84)
PLAN_SUCCESS_RATE_THRESHOLD: float = CONFIG.get("refinement", {}).get("plan_success_rate_threshold", 0.66)

# Import other modules
from utils import log_results
from metrics import compute_similarity, calculate_ew_score, evaluate_plan_success
from llm_manager import LLManager

# Environment class; assuming it's initialized outside and passed in
# It must have execute_plan(plan: str) -> bool and check_feasibility(seq: List[str], domain: str) -> bool

class Refinement:
    def __init__(self, environment, llm_manager: LLManager):
        """
        Initialize with environment object and LLManager instance.
        """
        self.env = environment
        self.llm_manager = llm_manager
        self.iteration: int = 0
        self.best_domain: str = ""
        self.best_ew_score: float = 0.0
        self.best_plan_success: bool = False
        self.success: bool = False
        self.planning_time: float = 0.0

    def run_refinement(self, initial_domain: str, initial_problem: str) -> Tuple[str, float, bool]:
        """
        Main refinement loop:
        - Iteratively generate refined domain models guided by EW scores
        - Evaluate plan success
        - Stop when thresholds are met or max iterations reached
        """
        start_time = time.time()

        # Initialize with initial domain
        current_domain = initial_domain
        # For logging best so far
        self.best_domain = initial_domain
        self.best_ew_score = 0.0
        self.best_plan_success = False
        self.iteration = 0
        self.success = False

        while self.iteration < MAX_REFINE_ITER:
            self.iteration += 1
            logging.info(f"Refinement iteration {self.iteration}")
            # Generate refined domain using LLM
            # Use current domain as context
            try:
                candidate_domain_str = self.llm_manager.refine_prompt(
                    current_domain=current_domain,
                    environment_feedback=self.best_ew_score
                )
            except Exception as e:
                logging.warning(f"LLM refinement call failed: {e}")
                # fallback: keep previous domain
                candidate_domain_str = current_domain

            # Validate syntax
            # PDDL validation could be improved, here we assume simple validation
            valid_syntax, _ = self.validate_pddl_syntax(candidate_domain_str)
            if not valid_syntax:
                logging.warning(f"Iteration {self.iteration}: Generated domain has invalid syntax, skipping refinement.")
                continue

            # Compute EW score
            try:
                ew_score = compute_similarity(
                    self.best_domain,
                    candidate_domain_str,
                    self.env
                )
            except Exception as e:
                logging.warning(f"EW computation failed at iteration {self.iteration}: {e}")
                ew_score = 0.0

            # Use domain that yields higher similarity
            if ew_score > self.best_ew_score:
                self.best_ew_score = ew_score

            # Check plan in environment
            plan = self.solve_in_environment(candidate_domain_str, initial_problem)
            if plan:
                plan_success = evaluate_plan_success(plan, self.env)
            else:
                plan_success = False

            # Update best if improved
            if ew_score >= EW_THRESHOLD and plan_success:
                logging.info(f"Iteration {self.iteration}: Accepting new domain with EW={ew_score:.3f} and plan success.")
                self.best_domain = candidate_domain_str
                self.best_plan_success = True
                # If thresholds are met, stop early
                if ew_score >= 1.0 or plan_success:
                    self.success = True
                    break
            else:
                logging.info(f"Iteration {self.iteration}: No improvement or thresholds not met. EW={ew_score:.3f}, Plan success={plan_success}")

            # Prepare for next iteration: possibly use best domain
            current_domain = self.best_domain

        total_time = time.time() - start_time
        self.planning_time = total_time

        # Final evaluation
        self.success = self.best_plan_success
        return self.best_domain, self.best_ew_score, self.success

    def solve_in_environment(self, domain_str: str, problem_str: str) -> Optional[str]:
        """
        Run classical planner on given domain/problem, return plan string if found.
        """
        # Save temporary files or pass strings if environment supports direct input
        # For simplicity, assume file paths or direct string inputs are used.
        # Here, use placeholder functions.
        try:
            # Prepare files (or pass strings directly if environment API supports)
            domain_path = "temp_domain.pddl"
            problem_path = "temp_problem.pddl"
            with open(domain_path, 'w') as f:
                f.write(domain_str)
            with open(problem_path, 'w') as f:
                f.write(problem_str)
            # Run planner
            from utils import plan_in_caller  # assume utility function
            plan_str = plan_in_caller(domain_path, problem_path)
            return plan_str
        except Exception as e:
            logging.warning(f"Planning failed: {e}")
            return None

    def validate_pddl_syntax(self, pddl_str: str) -> Tuple[bool, str]:
        """
        Validate PDDL syntax, here simplified as checking parse success.
        """
        import pddlpy
        try:
            if not pddl_str.strip().startswith('(define'):
                return False, pddl_str
            # Could add parser validation here
            return True, pddl_str
        except Exception:
            return False, pddl_str
```

## utils.py

```python
## utils.py

import os
import json
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union

# Load configuration parameters from config.yaml
import yaml

try:
    with open('config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    # Fallback default configuration if file not found
    CONFIG = {
        'environment': {
            'max_sampling_length': 10,
            'num_samples': 4
        },
        'refinement': {
            'max_refinement_iterations': 20,
            'ew_threshold': 0.84,
            'plan_success_rate_threshold': 0.66
        },
        'prompting': {
            'initial_prompt_template': "",
            'domain_prompt_template': "",
            'refinement_prompt_template': ""
        },
        'planner': {
            'classical_solver': "fast-downward",
            'plan_time_limit': 300
        },
        'logging': {
            'log_dir': "./logs",
            'log_level': "INFO",
            'save_resources': True
        }
    }

# Configure logging based on config.yaml
LOG_LEVEL = getattr(logging, CONFIG['logging'].get('log_level', 'INFO'))
LOG_DIR = CONFIG['logging'].get('log_dir', './logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=LOG_LEVEL,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(LOG_DIR, 'utils.log')),
                        logging.StreamHandler()
                    ])

def log_results(env_id: str, success: bool, ew_score: float, iterations: int, resource_usage: Dict[str, Union[int, float]]) -> None:
    """
    Logs the success status, EW score, number of refinement iterations, and resource usage details
    for each environment run to facilitate reproducibility and post-hoc analysis.
    """
    log_entry = {
        'environment_id': env_id,
        'success': success,
        'ew_score': ew_score,
        'iterations': iterations,
        'resource_usage': resource_usage,
        'timestamp': datetime.now().isoformat()
    }
    # Log at INFO level
    logging.info(f"Results for env {env_id}: {json.dumps(log_entry)}")
    # Save to a dedicated log file in JSON Line format for later analysis
    log_file_path = os.path.join(LOG_DIR, 'experiment_results.jsonl')
    try:
        with open(log_file_path, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
    except Exception as e:
        logging.error(f"Failed to write log results: {e}")

def generate_prompt_initial_problem(
    nl_description: str,
    object_list: List[str],
    api_signatures: Dict[str, Dict]
) -> str:
    """
    Constructs a prompt string for the LLM to generate an initial problem PDDL
    based on NL description, object list, and API signatures.
    """
    template = CONFIG['prompting'].get('initial_prompt_template', "")
    # Format object list as a string
    objects_str = "\n".join([f"{obj}" for obj in object_list])
    prompt = template.format(
        nl_description=nl_description,
        object_list=objects_str,
        api_signatures=json.dumps(api_signatures, indent=2)
    )
    return prompt

def generate_prompt_initial_domain(
    nl_description: str,
    api_signatures: Dict[str, Dict]
) -> str:
    """
    Constructs a prompt string for the LLM to generate a domain PDDL
    given NL description and API signatures.
    """
    template = CONFIG['prompting'].get('domain_prompt_template', "")
    prompt = template.format(
        nl_description=nl_description,
        api_signatures=json.dumps(api_signatures, indent=2)
    )
    return prompt

def refine_prompt(
    current_domain: str,
    environment_feedback: Union[str, float],
    additional_context: Optional[Dict[str, Union[str, float]]] = None
) -> str:
    """
    Constructs a refinement prompt to improve current domain model based on feedback and EW score.
    """
    template = CONFIG['prompting'].get('refinement_prompt_template', "")
    context_str = ""
    if additional_context:
        # Example: include previous EW, plan success metrics, etc.
        context_str = "\n".join([
            f"{key}: {value}" for key, value in additional_context.items()
        ])
    prompt = template.format(
        current_domain=current_domain,
        environment_feedback=str(environment_feedback),
        additional_context=context_str
    )
    return prompt

def execute_plan_in_environment(
    domain_file: str,
    problem_file: str,
    plan: str
) -> bool:
    """
    Submits a plan to the environment API for execution and returns success status.
    """
    # Placeholder: Replace with actual API call in environment.py
    # e.g.,
    # response = environment_api(interact=..., domain=domain_file, problem=problem_file, plan=plan)
    # success = response['success']
    # For illustration:
    success = False
    try:
        # Assume environment API is accessible and implemented
        from environment import Environment
        env = Environment(api_signatures={}, env_id='dummy')  # Placeholder
        success = env.execute_plan(plan)
    except Exception as e:
        logging.error(f"Error executing plan: {e}")
        success = False
    return success

def is_plan_executable_in_environment(
    domain: str,
    problem: str,
    action_sequence: List[str]
) -> bool:
    """
    Checks whether a sequence of actions is feasible in the environment without full execution.
    """
    try:
        from environment import Environment
        env = Environment(api_signatures={}, env_id='dummy')  # Placeholder
        feasible = env.check_feasibility(action_sequence)
        return feasible
    except Exception as e:
        logging.error(f"Error checking feasibility: {e}")
        return False

def sample_action_sequences(
    domain: str,
    problem: str,
    sequence_length: int,
    num_samples: int
) -> List[List[str]]:
    """
    Samples a set of diverse action sequences from the environment, constrained by T_max and num_samples.
    """
    sequences = []
    try:
        from environment import Environment
        env = Environment(api_signatures={}, env_id='dummy')  # Placeholder
        for _ in range(num_samples):
            seq = env.sample_action_sequence(sequence_length)
            sequences.append(seq)
    except Exception as e:
        logging.error(f"Error sampling sequences: {e}")
    return sequences

def compute_ew_score(
    domain1: str,
    domain2: str,
    sequences1: List[List[str]],
    sequences2: List[List[str]],
    environment: 'Environment',
    max_length: int
) -> float:
    """
    Calculates the symmetric EW score between domain1 and domain2 over sampled sequences.
    """
    def measure_similarity(domainA, domainB, seqsA, seqsB):
        total_score = 0.0
        for seq in seqsA:
            feasible = environment.check_feasibility(seq)  # in domainA
            if feasible:
                # Check if same sequence is executable in domainB
                exec_in_b = environment.check_feasibility(seq)
                total_score += float(exec_in_b)
        return total_score / len(seqsA) if seqsA else 0.0

    # Sample sequences if not provided
    if not sequences1:
        sequences1 = sample_action_sequences(domain1, '', max_length, CONFIG['environment']['num_samples'])
    if not sequences2:
        sequences2 = sample_action_sequences(domain2, '', max_length, CONFIG['environment']['num_samples'])

    score_d_dhat = measure_similarity(domain1, domain2, sequences1, sequences2)
    score_dhat_d = measure_similarity(domain2, domain1, sequences2, sequences1)

    # Harmonic mean for symmetry
    if score_d_dhat + score_dhat_d == 0:
        ew_score = 0.0
    else:
        ew_score = 2.0 * score_d_dhat * score_dhat_d / (score_d_dhat + score_dhat_d)
    return ew_score

def evaluate_plan_success(
    plan: str,
    environment: 'Environment'
) -> bool:
    """
    Checks whether the plan achieves environment goal.
    """
    try:
        success = environment.execute_plan(plan)
        # Alternatively, verify goal state via environment API
        return success
    except Exception as e:
        logging.error(f"Error evaluating plan success: {e}")
        return False

def parse_pddl_string(pddl_str: str) -> bool:
    """
    Utility to validate PDDL syntax correctness.
    """
    # Use a PDDL parser library if available, or basic syntax check
    # For simplicity, attempt to parse (mocked)
    try:
        import pddlpy
        # Can parse as a test, or invoke external parser
        # This is a placeholder; replace with actual parser call if available
        return True
    except Exception:
        return False

def load_resource_metrics(raw_output: Union[str, dict]) -> Dict[str, Union[int, float]]:
    """
    Extract resource usage data such as token counts, runtime, etc.
    """
    metrics = {}
    if isinstance(raw_output, dict):
        metrics = raw_output
    elif isinstance(raw_output, str):
        try:
            metrics = json.loads(raw_output)
        except json.JSONDecodeError:
            metrics = {}
    # Example: adding token count if available
    if 'tokens' in metrics:
        metrics['tokens'] = int(metrics['tokens'])
    if 'time' in metrics:
        metrics['time_seconds'] = float(metrics['time'])
    return metrics

def save_to_logfile(
    log_path: str,
    log_data: Dict
) -> None:
    """
    Saves logs and results as a JSON line for reproducibility.
    """
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, 'results_log.jsonl')
    try:
        with open(log_file, 'a') as f:
            json.dump(log_data, f)
            f.write('\n')
    except Exception as e:
        logging.error(f"Failed to save logs: {e}")

# Note:
# Several of these functions depend on environment.py for Environment class/methods.
# Implement environment accordingly, which manages actual environment interactions.
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\llm-pddl-planning\llm-pddl-planning_repo`
