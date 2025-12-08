## dataset_loader.py
import os
import json
import asyncio
import aiohttp
from typing import List, Optional
import numpy as np

# Define the ComparisonPair data structure
class ComparisonPair:
    def __init__(
        self,
        prompt_response_a: Optional[str],
        response_b: Optional[str],
        preference: int,
        label_objective: str = 'undefined',
        z: Optional[float] = None  # Hidden context variable
    ):
        self.prompt_response_a = prompt_response_a
        self.response_b = response_b
        self.preference = preference  # 1 if A preferred, 0 if B preferred
        self.label_objective = label_objective  # e.g., 'helpful', 'harmless'
        self.z = z  # latent hidden context, optional

class Dataset:
    def __init__(self, pairs: List[ComparisonPair]):
        self.pairs = pairs

class DatasetLoader:
    def __init__(self, dataset_type: str, relabel: bool):
        """
        Initialize with dataset type (e.g., 'synthetic', 'relabeled_hh_rlhf', 'real')
        and whether to relabel (for simulating divergence).
        """
        self.dataset_type = dataset_type
        self.relabel = relabel
        self.data: List[ComparisonPair] = []

        # Paths for preloaded datasets (placeholders)
        self.preloaded_data_path = 'data/preloaded_dataset.json'
        # For relabeling, cache responses
        self._gpt_cache = {}
        # GPT API details (replace with actual endpoint and API key)
        self.api_url = 'https://api.openai.com/v1/chat/completions'
        self.api_key = os.getenv('OPENAI_API_KEY', '')

    def load_data(self) -> Dataset:
        if self.dataset_type == 'synthetic':
            self.data = self.generate_synthetic_data()
        elif self.dataset_type == 'relabeled_hh_rlhf':
            # Load existing dataset
            self.data = self.load_preloaded_dataset()
            if self.relabel:
                # Run relabel asynchronously
                self.data = asyncio.run(self.relabel_data(self.data))
        elif self.dataset_type == 'real':
            # Load real dataset from file (placeholder)
            self.data = self.load_real_dataset()
            if self.relabel:
                self.data = asyncio.run(self.relabel_data(self.data))
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")
        return Dataset(self.data)

    def load_preloaded_dataset(self) -> List[ComparisonPair]:
        """
        Load dataset from a JSON file.
        Assumes a list of dicts with keys: 'prompt_response_a', 'response_b', 'preference', 'objective' (optional), 'z' (if stored)
        """
        if not os.path.exists(self.preloaded_data_path):
            raise FileNotFoundError(f"Preloaded dataset not found at {self.preloaded_data_path}")
        with open(self.preloaded_data_path, 'r') as f:
            data_json = json.load(f)
        pairs = []
        for entry in data_json:
            pairs.append(
                ComparisonPair(
                    prompt_response_a=entry.get('prompt_response_a'),
                    response_b=entry.get('response_b'),
                    preference=entry.get('preference'),
                    label_objective=entry.get('objective', 'unknown'),
                    z=entry.get('z', None)
                )
            )
        return pairs

    def load_real_dataset(self) -> List[ComparisonPair]:
        """
        Placeholder: Should load actual real dataset of preferences.
        For demonstration, load similar to preloaded and adapt.
        """
        # Replace with actual dataset loading code as needed
        # For now, simulate with empty or minimal data
        # Alternatively, load from CSV, JSONL, or other formats
        # Here, just raise exception to indicate placeholder
        raise NotImplementedError("Implement actual real dataset loading here.")

    def generate_synthetic_data(self) -> List[ComparisonPair]:
        """
        Generate synthetic alternatives and comparison pairs with known hidden context.
        """
        # Generate alternatives
        alternatives = np.linspace(0, 1, num=100, dtype=float)
        pairs = []
        num_pairs = 10000  # As per config or fixed
        for _ in range(num_pairs):
            a, b = np.random.choice(alternatives, size=2, replace=False)
            z = np.random.binomial(1, 0.5)  # hidden context
            u_a = self.true_utility(a, z)
            u_b = self.true_utility(b, z)
            preference = int(u_a > u_b)
            # Optional: generate prompt text or keep None
            prompt_text = None
            pairs.append(
                ComparisonPair(
                    prompt_response_a=prompt_text,
                    response_b=None,
                    preference=preference,
                    label_objective='synthetic',
                    z=z
                )
            )
        return pairs

    def true_utility(self, a: float, z: int) -> float:
        """
        True utility function: u(a,z)=a if a<0.8; else 2*a*z
        """
        if a < 0.8:
            return a
        else:
            return 2 * a * z

    async def relabel_data(self, dataset: List[ComparisonPair]) -> List[ComparisonPair]:
        """
        Use GPT API to re-annotate pairs according to a specified objective.
        For each pair, send conversational prompt to GPT and parse the response.
        """
        relabeled_pairs = []
        tasks = []
        for comp in dataset:
            prompt_text = self.build_relabel_prompt(comp, objective=comp.label_objective)
            tasks.append(self._call_gpt_comp(prompt_text))
        # Execute in parallel
        responses = await asyncio.gather(*tasks)
        for comp, resp_text in zip(dataset, responses):
            preference = self.parse_gpt_response(resp_text, comp)
            # For harmfulness relabeling, may invert preference
            if comp.label_objective == 'harmful':
                preference = 1 - preference
            relabeled_pairs.append(
                ComparisonPair(
                    prompt_response_a=comp.prompt_response_a,
                    response_b=comp.response_b,
                    preference=preference,
                    label_objective=comp.label_objective,
                    z=comp.z
                )
            )
        return relabeled_pairs

    def build_relabel_prompt(self, comp: ComparisonPair, objective: str) -> str:
        """
        Construct a prompt string to send to GPT for relabeling.
        Insert the comparison info into the template.
        """
        # Example prompt template (can be customized)
        if objective == 'helpful':
            prompt_template = (
                "In this task, you will be presented with a prompt and two responses from an AI assistant. "
                "Your task is to determine which response is more *helpful* and *honest*. "
                "Please consider carefully and select 'A' if the first response is more helpful/honest, or 'B' if the second is. "
                "Respond only with 'A' or 'B'.\n"
                "Prompt: <PROMPT_PLACEHOLDER>\n"
                "Response A: <RESPONSE_A_PLACEHOLDER>\n"
                "Response B: <RESPONSE_B_PLACEHOLDER>\n"
                "Which is more helpful and honest? (A/B):"
            )
        elif objective == 'harmful':
            prompt_template = (
                "In this task, you will be presented with a prompt and two responses from an AI assistant. "
                "Your task is to determine which response is *more harmful*. "
                "Please select 'A' or 'B' based on which is more harmful. Respond only with 'A' or 'B'.\n"
                "Prompt: <PROMPT_PLACEHOLDER>\n"
                "Response A: <RESPONSE_A_PLACEHOLDER>\n"
                "Response B: <RESPONSE_B_PLACEHOLDER>\n"
                "Which is more harmful? (A/B):"
            )
        else:
            prompt_template = "Unknown objective."

        prompt_text = prompt_template.replace('<PROMPT_PLACEHOLDER>', comp.prompt_response_a or '')
        prompt_text = prompt_text.replace('<RESPONSE_A_PLACEHOLDER>', comp.response_b or '')
        prompt_text = prompt_text.replace('<RESPONSE_B_PLACEHOLDER>', comp.response_b or '')
        return prompt_text

    async def _call_gpt_comp(self, prompt: str) -> str:
        """
        Call the GPT API asynchronously with caching.
        """
        # Check cache first
        if prompt in self._gpt_cache:
            return self._gpt_cache[prompt]
        # Compose payload
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'system', 'content': 'You are an AI assistant.'},
                         {'role': 'user', 'content': prompt}],
            'max_tokens': 1,
            'temperature': 0
        }
        # Send request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as resp:
                    resp_json = await resp.json()
                    message = resp_json['choices'][0]['message']['content']
        except Exception as e:
            # fallback or error handling
            message = 'A'  # default fallback
        self._gpt_cache[prompt] = message.strip()
        return message.strip()

    def parse_gpt_response(self, response_text: str, comp: ComparisonPair) -> int:
        """
        Parse GPT reply ('A' or 'B') and convert to preference label (1 or 0).
        """
        resp = response_text.lower()
        if 'a' in resp:
            return 1
        elif 'b' in resp:
            return 0
        else:
            # fallback to majority or default
            return 1  # default preference if ambiguous
