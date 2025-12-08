## reward_model.py

import os
import torch
import torch.nn.functional as F
from typing import List, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RewardModel:
    """
    Defines the RewardModel class to load a pretrained reward model, evaluate response scores
    for individual and batch responses given prompts, and provide interfaces for training and evaluation.
    Uses a classification or regression head based on the pretrained model.
    """

    def __init__(self, model_path: str = "path/to/reward/model", device: str = "cuda"):
        """
        Initializes the RewardModel:
        - Loads the pretrained reward model architecture and weights from model_path.
        - Loads the corresponding tokenizer.
        - Sets the model to evaluation mode.
        - Moves the model to the specified device.
        Args:
            model_path (str): Path or identifier for the pretrained reward model.
            device (str): 'cuda' or 'cpu', defaults to 'cuda' if available.
        """
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Check if model is a sequence classification or regression head
        # For simplicity, assume sequence classification with single label output
        # Users can adapt if using a different architecture
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()  # Set to eval mode for inference

    def score_response(self, prompt: str, response: str) -> float:
        """
        Scores a single response for a given prompt.
        Args:
            prompt (str): The input prompt.
            response (str): The response to score.
        Returns:
            float: Scalar reward score, higher indicates better response.
        """
        # Concatenate prompt and response or format input as needed
        input_text = f"Prompt: {prompt}\nResponse: {response}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Assume the model outputs logits with shape [batch_size, num_labels]
            logits = outputs.logits.squeeze(0)  # shape: [num_labels]
            # For binary classifier, logits[1] or logits[0] depending on class label
            # Assuming positive class score is at index 1
            # Also, if regression head, the output might be a single scalar
            if logits.shape[0] == 1:
                # Regression case
                score = logits.item()
            else:
                # Classification case
                # Use sigmoid on the positive class or max logit as proxy
                score = torch.sigmoid(logits[1]).item() if logits.shape[0] >=2 else logits[0].item()
        return score

    def score_responses(self, prompt: str, responses: List[str]) -> List[float]:
        """
        Scores a batch of responses for the same prompt.
        Args:
            prompt (str): The prompt string.
            responses (List[str]): List of responses to score.
        Returns:
            List[float]: List of scalar scores corresponding to responses.
        """
        # Prepare batch inputs
        input_texts = [
            f"Prompt: {prompt}\nResponse: {resp}"
            for resp in responses
        ]
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_batch = outputs.logits  # shape: [batch_size, num_labels]

        scores = []
        for logits in logits_batch:
            if logits.shape[0] == 1:
                # Regression case
                score = logits.item()
            else:
                # Classification case, sigmoid on positive class
                score = torch.sigmoid(logits[1]).item() if logits.shape[0] >= 2 else logits[0].item()
            scores.append(score)
        return scores

    # Optional: method to score multiple responses and prompts
    def score_batch(self, prompts: List[str], responses_list: List[List[str]]) -> List[List[float]]:
        """
        Scores many prompts with multiple responses each.
        Args:
            prompts (List[str]): List of prompts.
            responses_list (List[List[str]]): List of responses for each prompt.
        Returns:
            List[List[float]]: List of scores for each prompt's responses.
        """
        all_scores = []
        for prompt, responses in zip(prompts, responses_list):
            scores = self.score_responses(prompt, responses)
            all_scores.append(scores)
        return all_scores
