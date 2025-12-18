## model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

class ResponseGenerator:
    """
    Encapsulates a pretrained causal language model for token-level response generation,
    likelihood estimation, and checkpoint management.
    
    Supports sampling with diversity parameters and retrieval of token probability distributions.
    """
    def __init__(self, model_name: str = "gpt2-medium", checkpoint_path: Optional[str] = None):
        """
        Initialize the ResponseGenerator with a pretrained model and tokenizer.
        Optionally load weights from a checkpoint path.
        
        Args:
            model_name (str): Huggingface pretrained model identifier.
            checkpoint_path (Optional[str]): Path to a saved checkpoint to load weights from.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # If checkpoint provided, load checkpoint weights
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        # Default sampling parameters for generate_response
        self.default_generation_kwargs = {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id
        }

    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 1.0,
                          top_p: float = 0.95, top_k: int = 50) -> str:
        """
        Generate a response to the prompt using sampling strategies.
        
        Args:
            prompt (str): The input prompt string.
            max_tokens (int): Max tokens to generate beyond prompt.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling probability threshold.
            top_k (int): Top-k sampling parameter.
            
        Returns:
            str: Generated response string.
        """
        generation_kwargs = self.default_generation_kwargs.copy()
        generation_kwargs.update({
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        })

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **generation_kwargs)
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response

    def get_probability_distribution(self, tokens: torch.LongTensor, context: str):
        """
        Compute the probability distribution over vocabulary for a given token sequence,
        conditioned on the prompt + previous tokens.
        
        Args:
            tokens (torch.LongTensor): Token IDs for the tokens of interest.
            context (str): Prompt or previous sequence as string.
        
        Returns:
            torch.Tensor: Probability distribution over the vocab of shape [vocab_size].
        """
        # Encode the context + tokens to get input tensor
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        input_ids = torch.tensor([context_ids + tokens.tolist()], device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
        # Get the logits for the last position
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        return probs

    def save_checkpoint(self, filepath: str):
        """
        Save model weights and tokenizer to the specified directory.
        
        Args:
            filepath (str): Directory path to save model and tokenizer.
        """
        # Save model state
        torch.save(self.model.state_dict(), filepath + "/model.pt")
        # Save tokenizer
        self.tokenizer.save_pretrained(filepath)

    def load_checkpoint(self, filepath: str):
        """
        Load model weights and tokenizer from specified directory.
        
        Args:
            filepath (str): Directory containing saved weights and tokenizer.
        """
        self.model.load_state_dict(torch.load(filepath + "/model.pt", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        # Tokenizer is assumed to be saved via save_pretrained, so reload
        # optionally, but if needed, can reload tokenizer here.

