# model.py

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import os

class Model:
    """
    Encapsulates the text and image encoders (e.g., CLIP),
    provides methods to encode prompts and images, compute similarity,
    and load/save model weights.
    """

    def __init__(self, pretrained_model_name: str = "openai/clip-vit-base-patch32",
                 load_weights: str = None,
                 device: str = "cuda"):
        """
        Initializes the Model with pre-trained encoders.
        Optionally loads weights from a checkpoint.

        Args:
            pretrained_model_name (str): Name or path of the pretrained model.
            load_weights (str, optional): Path to a checkpoint weight file.
            device (str): 'cuda' or 'cpu'.
        """
        self.device = device
        # Load the processor and model from HuggingFace
        self.processor = CLIPProcessor.from_pretrained(pretrained_model_name)
        self.model = CLIPModel.from_pretrained(pretrained_model_name).to(self.device)
        self.model.eval()  # Set to eval mode for inference

        # Freeze all parameters for backbone unless fine-tuning is needed
        # Keep them trainable if fine-tuning later
        for param in self.model.parameters():
            param.requires_grad = False

        # Load weights if provided
        if load_weights:
            self.load_weights(load_weights)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """
        Encodes a prompt string into an embedding tensor.
        Uses the text encoder component of the model.
        """
        # Tokenize prompt with max 5 tokens
        inputs = self.processor(text=prompt, max_length=5, truncation=True, padding=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            output = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            # output shape: (batch_size=1, embed_dim)
            embedding = output.squeeze(0)
            # Normalize embedding
            embedding = nn.functional.normalize(embedding, p=2, dim=0)
        return embedding

    def encode_image(self, image_path: str) -> torch.Tensor:
        """
        Loads and preprocesses an image, encodes it into an embedding.
        """
        # Load image via PIL
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        # Process image
        inputs = self.processor(images=image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(self.device)

        with torch.no_grad():
            output = self.model.get_image_features(pixel_values=pixel_values)
            # output shape: (1, embed_dim)
            embedding = output.squeeze(0)
            # Normalize embedding
            embedding = nn.functional.normalize(embedding, p=2, dim=0)
        return embedding

    def compute_score(self, prompt_embedding: torch.Tensor, image_embedding: torch.Tensor) -> float:
        """
        Computes cosine similarity score between prompt and image embeddings.
        """
        # Since embeddings are normalized, dot product is cosine similarity
        score = torch.dot(prompt_embedding, image_embedding).item()
        return score

    def save_weights(self, path: str) -> None:
        """
        Saves the model's state_dict to the specified path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path: str) -> None:
        """
        Loads weights from the specified checkpoint file.
        Assumes the checkpoint is compatible with the current model.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
