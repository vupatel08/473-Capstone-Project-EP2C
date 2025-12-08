## model_api.py
import os
import requests
import logging
from typing import Optional, Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class ModelAPI:
    """
    This class provides an interface to interact with either an API-based large language model
    (e.g., OpenAI API) or a local HuggingFace transformers model.

    It supports deterministic decoding (temperature=0), handles prompt sending, and response parsing,
    following the design and configuration specifications.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        model_type: str = "local"
    ):
        """
        Initialize the ModelAPI instance.
        Args:
            model_name (str): Model identifier or local model path.
            api_key (Optional[str]): API key if using API-based models.
            temperature (float): Decoding temperature; default 0 for deterministic.
            max_new_tokens (int): Max tokens to generate.
            model_type (str): "api" for API-based, "local" for local models.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model_type = model_type.lower()

        if self.model_type == "api":
            # Assume using OpenAI API
            # Set API endpoint based on model name if necessary
            if self.model_name.lower().startswith("gpt-4") or self.model_name.lower().startswith("gpt-3.5"):
                self.api_url = "https://api.openai.com/v1/chat/completions"
            else:
                # If other API-based models, need to specify accordingly
                self.api_url = "https://api.openai.com/v1/completions"
            if self.api_key is None:
                raise ValueError("API key must be provided for API-based models.")
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        elif self.model_type == "local":
            # Initialize local model and tokenizer
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            except ImportError:
                raise ImportError("Transformers library required for local models.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            # Enable fast generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0,  # assume CUDA device 0; adjust as needed
                trust_remote_code=False
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def call_model(self, prompt: str, max_tokens: int) -> str:
        """
        Send the prompt to the model and return the generated output.
        Args:
            prompt (str): The input prompt string.
            max_tokens (int): Number of tokens to generate.

        Returns:
            str: The model's generated text response.
        """
        if self.model_type == "api":
            return self._call_api(prompt, max_tokens)
        elif self.model_type == "local":
            return self._call_local(prompt, max_tokens)
        else:
            raise RuntimeError(f"Unsupported model_type: {self.model_type}")

    def _call_api(self, prompt: str, max_tokens: int) -> str:
        """
        Call the OpenAI API with the prompt.
        """
        payload = {}
        if self.api_url.endswith("/chat/completions"):
            # Chat completion format
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "n": 1
            }
        else:
            # Text completion format
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "n": 1
            }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30  # seconds
            )
            response.raise_for_status()
            result_json = response.json()

            # Parse response depending on API format
            if self.api_url.endswith("/chat/completions"):
                # Chat API response
                reply = result_json['choices'][0]['message']['content']
            else:
                # Text completion API response
                reply = result_json['choices'][0]['text']
            return reply.strip()
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return f"Error: {str(e)}"
        except (KeyError, IndexError) as e:
            logging.error(f"Response parsing error: {e}")
            return f"Error: Response parsing error: {str(e)}"

    def _call_local(self, prompt: str, max_tokens: int) -> str:
        """
        Generate text from a local transformer model using pipeline with greedy decoding.
        """
        try:
            # Generate with greedy decoding (temperature=0)
            output = self.generator(
                prompt,
                max_length=len(prompt.split()) + max_tokens,
                do_sample=False,
                temperature=0.0,
                num_return_sequences=1,
                # Use top_k=50 if desired; here default suffice
            )
            # output is a list of dicts
            generated_text = output[0]['generated_text']
            # Remove the prompt from the generated text
            # Assumption: generated_text starts with prompt
            if generated_text.startswith(prompt):
                gen_response = generated_text[len(prompt):]
            else:
                gen_response = generated_text
            return gen_response.strip()
        except Exception as e:
            logging.error(f"Local model generation failed: {e}")
            return f"Error: {str(e)}"
