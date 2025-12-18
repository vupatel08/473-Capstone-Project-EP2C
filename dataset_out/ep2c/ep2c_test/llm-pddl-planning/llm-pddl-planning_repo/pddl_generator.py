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

