## model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional

class ModelWrapper:
    """
    A wrapper around Hugging Face transformer models for sequence generation,
    log probability computation, and detector score retrieval.
    """

    def __init__(self, model_name: str = "facebook/llama-2-7b-chat", device: str = "cuda"):
        """
        Initialize the ModelWrapper by loading the specified model and tokenizer.

        Args:
            model_name (str): Pretrained model identifier on Hugging Face.
            device (str): 'cuda' or 'cpu'.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.model_name = model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Set padding token if not set to avoid errors
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 120, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate a continuation for the given prompt.

        Args:
            prompt (str): Input prompt string.
            max_new_tokens (int): Max tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling probability threshold.

        Returns:
            str: Generated text.
        """
        # Encode prompt
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)

        # Generate response using model.generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode output
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Remove prompt part from generated output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text

    def log_prob(self, sequence: str, input_prompt: str) -> float:
        """
        Compute the log probability of sequence given input prompt.

        Args:
            sequence (str): The full sequence (prompt + response).
            input_prompt (str): The prompt string used to generate sequence.

        Returns:
            float: Log probability scalar.
        """
        # Tokenize prompt and sequence
        prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(self.device)
        seq_ids = self.tokenizer.encode(sequence, return_tensors='pt').to(self.device)

        # Concatenate prompt and sequence ids
        input_ids = torch.cat([prompt_ids, seq_ids], dim=1)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits  # shape: (1, total_seq_len, vocab_size)

        # Compute log probabilities
        # For each token in sequence (excluding prompt), compute probability conditioned on previous tokens
        # The prompt tokens are at the start; start index = prompt length
        prompt_len = prompt_ids.shape[1]
        seq_len = seq_ids.shape[1]
        total_log_prob = 0.0

        for i in range(prompt_len, prompt_len + seq_len):
            # The context is all tokens before token i
            input_ids_slice = input_ids[:, :i]
            with torch.no_grad():
                output = self.model(input_ids=input_ids_slice)
                logits_i = output.logits[:, -1, :]  # logits for the current token
                probs = torch.softmax(logits_i, dim=-1)
                token_id = input_ids[0, i]
                token_prob = probs[0, token_id]
                # Add log probability
                total_log_prob += torch.log(token_prob + 1e-12).item()

        return total_log_prob

    def get_score(self, sequence: str) -> float:
        """
        Obtain the detector score for the sequence.

        This method can be customized depending on the detector's interface.
        For simplicity, we assume a log-likelihood score or probability-based score.
        Here, for illustration, we return the mean token probability as a proxy.
        In practice, this would interface with an external detector.

        Args:
            sequence (str): Text sequence to score.

        Returns:
            float: 'Human-ness' score, higher indicates more human-like.
        """
        # For this implementation, assume a placeholder score
        # E.g., a language model probability or detector API call
        # Here, return a dummy value; replace with actual detector API call if available
        # For example purposes, using negative perplexity as score:
        # negative_perplexity can be used; or integrate with detector API

        # Placeholder: assuming higher score means more human
        # For realistic scenario, replace with detector API call
        return 0.5  # Placeholder score

    def get_log_prob(self, sequence: str, input_prompt: str) -> float:
        """
        Compute and return the log probability of the sequence conditioned on the prompt.

        Args:
            sequence (str): The output sequence.
            input_prompt (str): The input prompt.

        Returns:
            float: Log-likelihood score.
        """
        return self.log_prob(sequence, input_prompt)
