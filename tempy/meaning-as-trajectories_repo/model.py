## model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import numpy as np

class ModelWrapper:
    """
    A wrapper class that handles loading a pre-trained autoregressive language model,
    tokenization, sequence generation (sampling), and likelihood computation.
    Supports models like GPT-2, LLaMA, Falcon, and extended to multimodal models like LLaVA,
    assuming they provide appropriate tokenization and inference APIs.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        """
        Initialize the ModelWrapper with the specified model.
        Args:
            model_name (str): HuggingFace model identifier.
            device (str): 'cuda' or 'cpu'.
        """
        self.model_name = model_name
        self.device = device

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # For models supporting causal LM:
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Set special token IDs
        # If model doesn't have an eos_token, we set it to tokenizer.eos_token
        # Typically, GPT-like models already have eos_token
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
            self.eos_token = self.tokenizer.eos_token
            self.eos_token_id = self.tokenizer.eos_token_id
        else:
            # Fallback: create a special [EOS] token if not present
            self.eos_token = None
            self.eos_token_id = None

        # Additional model-specific settings could go here

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes input text into token IDs.
        Args:
            text (str): Input string.
        Returns:
            List[int]: Token IDs.
        """
        # For general models
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return tokens

    def detokenize(self, tokens: List[int]) -> str:
        """
        Convert token IDs back to string.
        Args:
            tokens (List[int]): Token IDs.
        Returns:
            str: Decoded text.
        """
        return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=True)

    def generate_continuations(self, prompt: str, n: int =1, max_length: int=20, temperature: float=1.0) -> List[str]:
        """
        Generate n continuations (trajectories) for the given prompt.
        Uses stochastic sampling with temperature.
        Args:
            prompt (str): Input prompt.
            n (int): Number of trajectories.
            max_length (int): Max tokens in each continuation.
            temperature (float): Sampling temperature.
        Returns:
            List[str]: List of generated continuation strings.
        """
        generated_sequences: List[str] = []

        # Tokenize prompt
        input_ids = self.tokenize(prompt)
        input_ids = torch.tensor([input_ids], device=self.device)

        # For each trajectory
        for _ in range(n):
            # Prepare input
            input_ids_curr = input_ids.clone()
            # Generate sequence step-by-step
            for _ in range(max_length):
                outputs = self.model(input_ids=input_ids_curr)
                logits = outputs.logits  # shape: (1, seq_len, vocab_size)
                # Take the logits of the last token
                next_token_logits = logits[0, -1, :]  # shape: (vocab_size,)

                # Apply temperature
                probs = torch.softmax(next_token_logits / temperature, dim=-1)

                # Sample next token
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                # Append to sequence
                input_ids_curr = torch.cat([input_ids_curr, torch.tensor([[next_token_id]], device=self.device)], dim=-1)

                # Check for [EOS]
                if self.eos_token_id is not None and next_token_id == self.eos_token_id:
                    break
            # Decode generated tokens
            gen_tokens = input_ids_curr[0].tolist()
            # Remove the prompt tokens for output, or return full sequence (following the paper, include full sequence)
            # Here, per the paper, trajectories are continuations after prompt
            generation = self.detokenize(gen_tokens)
            # Optionally, strip prompt part. But here, trajectory includes prompt, aligning with the paper's measure.
            generated_sequences.append(generation)
        return generated_sequences

    def compute_log_prob(self, sequence: str, prompt: str) -> float:
        """
        Computes the log probability of a sequence conditioned on the prompt.
        Args:
            sequence (str): The continuation sequence.
            prompt (str): The prompt string.
        Returns:
            float: Log likelihood of the sequence given the prompt.
        """
        # Tokenize prompt and sequence
        prompt_ids = self.tokenize(prompt)
        sequence_ids = self.tokenize(sequence)

        # Concatenate prompt and sequence tokens
        input_ids = prompt_ids + sequence_ids
        input_ids_tensor = torch.tensor([input_ids], device=self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids_tensor)
            logits = outputs.logits  # shape: (1, total_seq_len, vocab_size)

        # Convert to probabilities
        # For likelihood of the sequence, compute the probability of each token conditioned on previous tokens.
        log_probs = []

        # Iterate over sequence tokens (after prompt)
        for i in range(len(prompt_ids), len(input_ids)):
            # Get logits for position i
            logits_i = logits[0, i -1, :]  # logits for token i conditioned on previous
            probs = torch.softmax(logits_i, dim=-1)
            token_id = input_ids[i]
            prob = probs[token_id]
            # Avoid log(0)
            if prob <= 0:
                prob = 1e-12
            log_probs.append(torch.log(prob))

        # Sum log probabilities
        total_log_prob = sum(log_probs).item()
        return total_log_prob
