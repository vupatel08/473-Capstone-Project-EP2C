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
