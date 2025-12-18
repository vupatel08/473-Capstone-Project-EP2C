## discriminator.py
import os
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ResponseDataset(Dataset):
    """
    Dataset for storing prompt-response pairs with labels for discriminator training.
    """
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        """
        Initialize dataset with data.

        Args:
            data (List[Dict]): List of dicts, each with 'prompt', 'response', 'label'.
            tokenizer: Huggingface tokenizer.
            max_length (int): Max sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        for item in data:
            # Concatenate prompt and response with separator
            full_text = item['prompt'] + self.tokenizer.eos_token + item['response']
            encodings = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding='max_length')
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']
            label = 1 if item['label'] == 'human' else 0
            self.samples.append({
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask),
                'label': torch.tensor(label, dtype=torch.float)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Discriminator:
    def __init__(self, config: Dict):
        """
        Initialize the lightweight discriminator classifier.

        Args:
            config (Dict): Configuration dictionary from 'config.yaml'.
        """
        model_name_or_path: str = config['model'].get('pretrained_model_name_or_path', 'gpt2-medium')
        # Load lightweight classification model; for simplicity, use a sequence classification model
        self.device = torch.device('cuda' if torch.has_cuda and config.get('use_gpu', True) else 'cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=1,  # Output scalar score
            output_attentions=False,
            output_hidden_states=False
        )
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Ensure tokenizer has pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Loss function for binary classification
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = None  # Will be initialized during training

    def train(self, dataset: List[Dict], epochs: int = 3, batch_size: int = 32, learning_rate: float = 1e-4):
        """
        Train discriminator on prompt-response pairs with labels.

        Args:
            dataset (List[Dict]): List of dicts with 'prompt', 'response', 'label' ('human' or 'model').
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            learning_rate (float): Learning rate.
        """
        # Prepare dataset
        train_dataset = ResponseDataset(dataset, self.tokenizer, max_length=512)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].unsqueeze(1).to(self.device)  # Shape (B,1)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)  # Shape (B,)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Discriminator Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (batch_idx+1):.4f}")

    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        """
        Compute scalar scores for each prompt-response pair indicating "human-likeness".

        Args:
            prompts (List[str]): List of prompts.
            responses (List[str]): Corresponding responses.

        Returns:
            List[float]: List of scores (e.g., scalar logits).
        """
        self.model.eval()
        scores = []
        with torch.no_grad():
            for prompt, response in zip(prompts, responses):
                full_text = prompt + self.tokenizer.eos_token + response
                encodings = self.tokenizer(full_text, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                score = logits.squeeze().item()  # Scalar score (logit)
                scores.append(score)
        return scores

    def save(self, checkpoint_path: str):
        """
        Save the discriminator model to the checkpoint path.

        Args:
            checkpoint_path (str): Directory to save model.
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

    def load(self, checkpoint_path: str):
        """
        Load discriminator model from checkpoint.

        Args:
            checkpoint_path (str): Path to saved model directory.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
