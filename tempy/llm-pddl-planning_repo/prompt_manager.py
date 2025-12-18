## prompt_manager.py

import yaml
from typing import Dict, Tuple

class PromptManager:
    """
    Manages prompt templates and generates prompts for PDDL generation and refinement.
    Loads prompt templates from configuration and provides methods to produce
    context-specific prompts for the LLM.
    """
    def __init__(self, templates: Dict[str, str]):
        """
        Initializes the PromptManager with provided prompt templates.
        Args:
            templates (Dict[str, str]): Dictionary containing templates:
                - 'initial_prompt_template'
                - 'refinement_prompt_template'
                - 'problem_prompt_template'
        """
        self.initial_prompt_template: str = templates.get('initial_prompt_template', "")
        self.refinement_prompt_template: str = templates.get('refinement_prompt_template', "")
        self.problem_prompt_template: str = templates.get('problem_prompt_template', "")

    def generate_initial_prompt(self, domain_NL: str, problem_NL: str) -> str:
        """
        Generates the prompt to produce initial PDDL domain and problem files.
        Args:
            domain_NL (str): Natural language description of the domain.
            problem_NL (str): Natural language description of the task/problem.
        Returns:
            str: The formatted prompt string for initial PDDL generation.
        """
        prompt = self.initial_prompt_template.format(
            domain_NL=domain_NL.strip(),
            problem_NL=problem_NL.strip()
        )
        return prompt

    def generate_refinement_prompt(self, prev_d: str, prev_p: str, feedback: str, env_feedback: str) -> str:
        """
        Generates the prompt for refining existing PDDL files based on environment feedback.
        Args:
            prev_d (str): Previous PDDL domain code.
            prev_p (str): Previous PDDL problem code.
            feedback (str): Natural language feedback or response from previous attempt.
            env_feedback (str): Environment exploration walk results, errors, or execution info.
        Returns:
            str: The crafted prompt for GPT-4 to improve PDDL models.
        """
        prompt = self.refinement_prompt_template.format(
            previous_domain=prev_d.strip(),
            previous_problem=prev_p.strip(),
            environment_feedback=env_feedback.strip(),
            model_feedback=feedback.strip()
        )
        return prompt

    def generate_problem_prompt(self, pddl_template: str, env_objects: list, nl_task: str) -> str:
        """
        Generates a prompt to produce a problem PDDL file based on environment objects and NL task.
        Args:
            pddl_template (str): A template or base snippet for the problem PDDL.
            env_objects (list): List of environment object identifiers.
            nl_task (str): Natural language description of the task and goals.
        Returns:
            str: The prompt string instructing to generate a valid problem PDDL.
        """
        objects_str = "\n".join(sorted(env_objects))
        prompt = self.problem_prompt_template.format(
            pddl_template=pddl_template.strip(),
            objects=objects_str,
            task_description=nl_task.strip()
        )
        return prompt
