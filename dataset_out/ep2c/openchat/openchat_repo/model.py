## model.py
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    """
    Encapsulates a pre-trained LLaMA-based language model with class-conditioning support.
    Provides methods for forward inference (training) and response generation.
    """

    def __init__(self, pretrained_model_name: str = "huggingface/llama-13b", conditioning_token: str = "<|class|>"):
        """
        Load the pre-trained model and tokenizer, set up special tokens and device.

        Args:
            pretrained_model_name (str): Hugging Face model identifier.
            conditioning_token (str): Special token used for class conditioning.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conditioning_token = conditioning_token

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name).to(self.device)

        # Ensure conditioning token is in tokenizer
        if self.conditioning_token not in self.tokenizer.get_vocab():
            # Add special conditioning token
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.conditioning_token]})
            # Resize model embeddings accordingly
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.conditioning_token_id = self.tokenizer.convert_tokens_to_ids(self.conditioning_token)

        # For convenience, set to evaluation mode
        self.model.eval()

        # Optional: store a cache of class labels to prompt prefixes
        self.class_prefixes = {
            "expert": f"{self.conditioning_token} GPT4 User:",
            "sub_optimal": f"{self.conditioning_token} GPT3 User:",
            # Add more if needed
        }

    def set_conditioning(self, class_label: str) -> str:
        """
        Retrieve or create the prompt prefix for a given class label.

        Args:
            class_label (str): e.g., "expert" or "sub_optimal"

        Returns:
            prompt_prefix (str): conditioned prefix
        """
        prefix = self.class_prefixes.get(class_label, f"{self.conditioning_token} User:")
        return prefix

    def prepare_prompt(self, prompt: str, class_label: Optional[str] = None) -> torch.Tensor:
        """
        Construct the conditioned input prompt sequence.

        Args:
            prompt (str): user instruction or dialogue turn
            class_label (str, optional): class label for conditioning

        Returns:
            input_ids (torch.Tensor): tokenized input sequence
        """
        if class_label:
            prefix = self.set_conditioning(class_label)
            full_prompt = f"{prefix}\n{prompt}"
        else:
            # If no class label provided, use a default prefix
            full_prompt = prompt

        # Encode the prompt
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        return input_ids

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Run the model forward pass for training. Returns logits.

        Args:
            input_ids (torch.Tensor): input sequence
            attention_mask (torch.Tensor, optional): attention mask

        Returns:
            logits (torch.Tensor): output logits, shape (batch_size, seq_len, vocab_size)
        """
        # Model expects batch dimension; assume batch size=1 here
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits  # shape: (1, seq_len, vocab_size)
        return logits

    def generate(
        self,
        prompt: str,
        class_label: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """
        Generate a response conditioned on the prompt and optional class label.

        Args:
            prompt (str): user input prompt
            class_label (str, optional): conditioning class label
            max_new_tokens (int): maximum tokens to generate
            temperature (float): sampling temperature
            do_sample (bool): whether to sample or greedy decode
            top_p (float): nucleus sampling probability
            top_k (int): top-k sampling

        Returns:
            response (str): generated text response
        """
        # Prepare conditioned prompt
        conditioned_prompt = self.prepare_prompt(prompt, class_label)

        # Generate response
        output_ids = self.model.generate(
            input_ids=conditioned_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=1,
        )

        # Decode output
        decoded_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract the reply part: remove the prompt leading part
        # Here, we naive strip the prompt part out, assuming generate appends to prompt
        response = decoded_text[len(self.tokenizer.decode(conditioned_prompt[0], skip_special_tokens=True)) :].strip()

        return response

    def save(self, save_path: str):
        """
        Save model and tokenizer to disk.

        Args:
            save_path (str): directory path to save the model
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load(self, load_path: str):
        """
        Load model and tokenizer from disk.

        Args:
            load_path (str): directory path where model is saved
        """
        self.model = AutoModelForCausalLM.from_pretrained(load_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
