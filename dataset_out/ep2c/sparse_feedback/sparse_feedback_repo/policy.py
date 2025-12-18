# policy.py

from typing import List
from dataset_loader import LanguageModel
from reward_model import RewardModel

class PolicySampler:
    """
    Implements the Best-of-n policy: generates multiple candidate responses
    for an instruction, scores them with a reward model, and returns the top response.
    """
    def __init__(
        self,
        language_model: LanguageModel,
        reward_model: RewardModel,
        n_responses: int = 64,
        temperature: float = 0.0
    ):
        """
        Args:
            language_model (LanguageModel): instance for response generation
            reward_model (RewardModel): instance for scoring responses
            n_responses (int): number of responses to generate per instruction
            temperature (float): sampling temperature for generation
        """
        self.language_model = language_model
        self.reward_model = reward_model
        self.n_responses = n_responses
        self.temperature = temperature

    def sample_responses(self, instruction: str) -> List[str]:
        """
        Generate multiple responses for a given instruction.
        Returns:
            responses (List[str]): list of generated responses
        """
        responses = self.language_model.generate(
            prompt=instruction,
            max_length=128,
            temperature=self.temperature,
            num_return_sequences=self.n_responses
        )
        return responses

    def score_responses(self, responses: List[str], instruction: str) -> List[float]:
        """
        Score each response using the reward model.
        Returns:
            scores (List[float]): list of scalar scores (higher is better)
        """
        scores = []
        for resp in responses:
            score = self.reward_model.score_response(resp, instruction)
            scores.append(score)
        return scores

    def select_best_response(self, responses: List[str], scores: List[float]) -> str:
        """
        Select the response with the highest score.
        """
        max_idx = scores.index(max(scores))
        return responses[max_idx]

    def generate_and_select(self, instruction: str) -> str:
        """
        Complete pipeline: generate multiple responses, score, and pick the best.
        Returns:
            best_response (str)
        """
        responses = self.sample_responses(instruction)
        scores = self.score_responses(responses, instruction)
        best_response = self.select_best_response(responses, scores)
        return best_response
