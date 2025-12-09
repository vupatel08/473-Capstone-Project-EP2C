## label_verifier.py
import os
import re
import json
import logging
from typing import Optional
import openai
from utils import load_config, get_prompt

logger = logging.getLogger(__name__)

class LabelVerifier:
    """
    The LabelVerifier class implements correctness verification of generated reasoning
    and answers using a structured prompt to a code-capable LLM (e.g., GPT-4 with code interpreter).
    It determines whether the generated label (answer) matches the inferred correctness based
    on reasoning steps, as per structured prompts defined in 'verification_prompt.txt'.
    """

    def __init__(self, api_key: str):
        """
        Initialize the verifier with API key, load config, set prompt template.
        Args:
            api_key (str): API key for OpenAI.
        """
        self.api_key = api_key
        # Load configuration
        self.config = load_config()
        # Set OpenAI API key
        openai.api_key = self.api_key
        # Load prompt template for verification
        self.prompt_template = get_prompt('label_verification')
        # API parameters
        self.temperature = self.config.get('model', {}).get('temperature', 0.0)
        self.max_tokens = self.config.get('model', {}).get('max_tokens', 1024)
        # Retry limit for robustness
        self.max_retries = 3

    def verify_label(self, generated_text: str) -> bool:
        """
        Verify the correctness of the generated reasoning and answer.
        Args:
            generated_text (str): The text output from the data generator, including reasoning, answer.
        Returns:
            bool: True if verified as correct, False otherwise.
        """
        prompt = self._construct_verification_prompt(generated_text)
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.config.get('model', {}).get('name', 'gpt-4'),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=0.95,
                    messages=[
                        {"role": "system", "content": "You are an assistant that verifies reasoning correctness."},
                        {"role": "user", "content": prompt}
                    ],
                )
                reply = response.choices[0].message['content'].strip()
                logger.info(f"Verification attempt {attempt + 1}: {reply}")
                return self._parse_verification_response(reply)
            except Exception as e:
                logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
        # If all retries fail, default to False
        logger.error("Verification failed after retries.")
        return False

    def _construct_verification_prompt(self, text: str) -> str:
        """
        Insert the generated reasoning-answer text into the verification template.
        Args:
            text (str): The model-generated text containing reasoning and answer.
        Returns:
            str: Structured prompt for verification.
        """
        prompt = self.prompt_template.replace("{generated_text}", text)
        return prompt

    def _parse_verification_response(self, response_text: str) -> bool:
        """
        Parse the model's response to determine correctness.
        Looks for keywords like 'Correct', 'Incorrect', 'Yes', 'No'.
        Args:
            response_text (str): The raw output from the model.
        Returns:
            bool: True if the response indicates correctness, False otherwise.
        """
        # Normalize response to lowercase for keyword search
        lowered = response_text.lower()
        # Check for explicit correctness indicators
        if 'correct' in lowered:
            return True
        if 'incorrect' in lowered:
            return False
        # Additional heuristic: yes/no
        if re.search(r'\b(yes|correct)\b', lowered):
            return True
        if re.search(r'\b(no|incorrect)\b', lowered):
            return False
        # Default fallback: if uncertain, return False
        return False
