## model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

class Model:
    def __init__(self, pretrained_model_path: str, embedding_dim: int = 768, segment_size: int = 96):
        """
        Initialize the AutoTimes model with a pre-trained decoder-only language model.
        Args:
            pretrained_model_path (str): Name or path of the pretrained LM (e.g., 'LLaMA-7B', 'gpt2', 'facebook/opt-1.3b').
            embedding_dim (int): Dimension of the LM token embeddings (D).
            segment_size (int): Number of data points per segment token (S).
        """
        # Load the pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.lm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)
        self.lm_model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lm_model.to(self.device)

        # Freeze all LM parameters
        for param in self.lm_model.parameters():
            param.requires_grad = False

        # Store configuration
        self.D = embedding_dim
        self.S = segment_size

        # Initialize segmentation embedding (MLP): input -> D
        self.segment_embed = nn.Sequential(
            nn.Linear(self.S, 512),
            nn.ReLU(),
            nn.Linear(512, self.D)
        )
        # Initialize projection head for decoding token embeddings back to series segments
        self.segment_projection = nn.Sequential(
            nn.Linear(self.D, 512),
            nn.ReLU(),
            nn.Linear(512, self.S)
        )

        # Set trainable parameters: only segment_embed and segment_projection
        for param in self.segment_embed.parameters():
            param.requires_grad = True
        for param in self.segment_projection.parameters():
            param.requires_grad = True

        # Use the lm's embedding layer for timestamp text
        self.text_token_embedding = self.lm_model.transformer.wte

    def freeze_backbone(self):
        """
        Freeze parameters of the LM backbone.
        """
        for param in self.lm_model.parameters():
            param.requires_grad = False
        # Unfreeze trainable layers if needed (by default, only embed/ proj are trainable)

    def embed_segments(self, segments: torch.Tensor) -> torch.Tensor:
        """
        Embed raw series segments into the model's latent space.
        Args:
            segments (torch.Tensor): shape (batch_size, S), raw series data.
        Returns:
            torch.Tensor: shape (batch_size, D), embedded segment vectors.
        """
        # segments shape: (B, S)
        embedded = self.segment_embed(segments)
        return embedded

    def embed_timestamps(self, timestamps: List[str]) -> torch.Tensor:
        """
        Convert timestamp strings into timestamp embeddings.
        Args:
            timestamps (List[str]): list of textual timestamps.
        Returns:
            torch.Tensor: shape (len(timestamps), D), timestamp embeddings.
        """
        # Tokenize text timestamps
        tokens = self.tokenizer(timestamps, return_tensors='pt', padding=True, truncation=True)
        input_ids = tokens['input_ids'].to(self.device)  # (L, T)
        # Obtain token embeddings
        with torch.no_grad():
            token_embeds = self.text_token_embedding(input_ids)  # (L, T, D)
        # Extract the embedding for <EOS> token, assuming tokenizer.eos_token_id exists
        eos_token_id = self.tokenizer.eos_token_id
        # Find index of eos_token in input_ids to get <EOS> embedding
        eos_indices = (input_ids == eos_token_id).nonzero(as_tuple=True)
        # Gather embeddings at eos positions
        eos_embeds = token_embeds[eos_indices[0], eos_indices[1], :]  # shape (L, D)
        # If multiple timestamps, return tensor
        return eos_embeds

    def embed_input(self, series_segments: torch.Tensor, timestamp_embeds: torch.Tensor) -> torch.Tensor:
        """
        Combine series segment embeddings and timestamp embeddings to form input token embeddings.
        Args:
            series_segments (torch.Tensor): shape (B, S), raw data.
            timestamp_embeds (torch.Tensor): shape (B, D)
        Returns:
            torch.Tensor: shape (B, D), combined embeddings.
        """
        segment_embeds = self.embed_segments(series_segments)  # (B, D)
        # Add timestamp embedding (broadcast over batch)
        input_embeddings = segment_embeds + timestamp_embeds
        return input_embeddings

    def predict_next_embeddings(self, input_embeddings: torch.Tensor, max_new_tokens: int = 1) -> torch.Tensor:
        """
        Given input sequence embeddings, predict next token embeddings autoregressively.
        Args:
            input_embeddings (torch.Tensor): shape (seq_len, D)
            max_new_tokens (int): how many tokens to generate
        Returns:
            torch.Tensor: predicted token embeddings for next tokens, shape (max_new_tokens, D)
        """
        seq_embeddings = input_embeddings.unsqueeze(0)  # (1, seq_len, D)
        generated_embeddings: List[torch.Tensor] = []

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.lm_model(inputs_embeds=seq_embeddings)
                logits = outputs.logits  # shape (1, seq_len, vocab_size)
            # Use last token's logits
            last_logits = logits[:, -1, :]  # (1, vocab_size)
            # For simplicity, pick argmax token
            next_token_id = torch.argmax(last_logits, dim=-1)  # (1,)
            # Convert token id to embedding
            next_token_embed = self.text_token_embedding(next_token_id)  # (1, D)
            generated_embeddings.append(next_token_embed.squeeze(0))
            # Append next token embed to sequence for next iteration
            seq_embeddings = torch.cat([seq_embeddings, next_token_embed.unsqueeze(0)], dim=1)  # (1, seq+1, D)

        return torch.stack(generated_embeddings, dim=0)  # (max_new_tokens, D)

    def decode_tokens(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Convert predicted token embeddings back into series segments.
        Args:
            token_embeddings (torch.Tensor): shape (T, D)
        Returns:
            torch.Tensor: shape (T, S), series segments.
        """
        # Map embeddings to series space
        series_segments = self.segment_projection(token_embeddings)  # (T, S)
        return series_segments

    def forward(self, input_series: torch.Tensor, timestamps: List[str], target_horizon: int = 96) -> torch.Tensor:
        """
        Forward pass for training: given input series and timestamps, predict the next tokens.
        Args:
            input_series (torch.Tensor): shape (B, S), current series segments.
            timestamps (List[str]): list of textual timestamp strings for the batch.
            target_horizon (int): number of segments to predict.
        Returns:
            torch.Tensor: decoded series segments, shape (B, target_horizon, S).
        """
        # Embed input series
        timestamp_embeds = self.embed_timestamps(timestamps)  # (B, D)
        input_embeds = self.embed_input(input_series, timestamp_embeds)  # (B, D)
        input_embeds = input_embeds  # (B, D)

        # Initialize sequence
        seq_embeds = input_embeds.unsqueeze(0)  # (1, B, D)
        outputs: List[torch.Tensor] = []

        # Generate target_horizon segments autoregressively
        for _ in range(target_horizon):
            with torch.no_grad():
                outputs_logits = self.lm_model(inputs_embeds=seq_embeds)  # (1, seq, vocab_size)
                last_logits = outputs_logits.logits[:, -1, :]  # (1, vocab_size)
            # Select argmax token
            next_token_id = torch.argmax(last_logits, dim=-1)  # (1,)
            # Map to embedding
            next_embed = self.text_token_embedding(next_token_id)  # (1, D)
            # Append to sequence
            seq_embeds = torch.cat([seq_embeds, next_embed.unsqueeze(1)], dim=1)  # (1, seq+1, D)
            outputs.append(next_embed.squeeze(0))

        # Stack predicted token embeddings
        predicted_token_embeds = torch.stack(outputs, dim=0)  # (target_horizon, D)
        # Decode into series segments
        predicted_series_segments = self.decode_tokens(predicted_token_embeds)  # (T, S)
        # Reshape to (target_horizon, S)
        predicted_series_segments = predicted_series_segments.reshape(-1, self.S)
        return predicted_series_segments

    def inference(self, last_series_segment: torch.Tensor, timestamp: str, predict_steps: int) -> torch.Tensor:
        """
        Autoregressive inference: generate multiple future segments.
        Args:
            last_series_segment (torch.Tensor): last known series segment, shape (S,)
            timestamp (str): textual timestamp for the next step
            predict_steps (int): number of segments to generate
        Returns:
            torch.Tensor: predicted series segments, shape (predict_steps, S)
        """
        generated_segments: List[torch.Tensor] = []
        current_segment = last_series_segment.unsqueeze(0)  # shape (1, S)
        timestamp_emb = self.embed_timestamps([timestamp])  # shape (1, D)

        # Initialize embedded sequence
        input_embeds = self.embed_input(current_segment, timestamp_emb)  # (1, D)

        seq_embeds = input_embeds.unsqueeze(0)  # (1, 1, D)

        for _ in range(predict_steps):
            with torch.no_grad():
                outputs_logits = self.lm_model(inputs_embeds=seq_embeds)  # (1, seq, vocab_size)
                last_logits = outputs_logits.logits[:, -1, :]  # (1, vocab_size)
            next_token_id = torch.argmax(last_logits, dim=-1)  # (1,)
            next_embed = self.text_token_embedding(next_token_id)  # (1, D)
            seq_embeds = torch.cat([seq_embeds, next_embed.unsqueeze(1)], dim=1)  # append new token
            # Decode token embedding into series segment
            series_token = self.decode_tokens(next_embed)  # (1, S)
            generated_segments.append(series_token.squeeze(0))
            # Update last_series_segment for next prediction if needed
            current_segment = series_token.squeeze(0)

        return torch.stack(generated_segments, dim=0)  # (predict_steps, S)
