## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import logging
import os

from dataset_loader import DatasetLoader
from model import Model
from utils import plot_prompt_clusters, plot_ranking_visualization, set_seed

import yaml

# Load configuration from 'config.yaml'
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device(config['training'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

# Set global seed for reproducibility
set_seed(config['training'].get('seed', 42))


class Trainer:
    """
    The Trainer class orchestrates training, validation, and evaluation of the Social Reward model
    following the methodology described in the paper and design specifications.
    """

    def __init__(self,
                 model: Model,
                 train_dataset: List[Dict],
                 val_dataset: List[Dict],
                 test_dataset: List[Dict],
                 config: dict = None):
        """
        Initialize the trainer with model, datasets, and hyperparameters.

        Args:
            model (Model): The social reward model to be trained/fined.
            train_dataset (list): List of training triplets.
            val_dataset (list): List of validation triplets.
            test_dataset (list): List of test triplets.
            config (dict): Configuration parameters.
        """
        self.model = model
        self.train_data = train_dataset
        self.val_data = val_dataset
        self.test_data = test_dataset
        self.config = config if config is not None else {}
        # Hyperparameters
        self.learning_rate = self.config['training'].get('learning_rate', 3e-4)
        self.batch_size = self.config['training'].get('batch_size', 32)
        self.epochs = self.config['training'].get('epochs', 10)
        self.margin = self.config['loss'].get('margin', 0.2)
        self.device = torch.device(self.config['training'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

        # Prepare optimizer - only parameters that require grads
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.model.parameters()), lr=self.learning_rate)

        # Placeholder for best validation accuracy
        self.best_val_accuracy = 0.0

        # Initialize DataLoaders
        self.train_loader = self._create_dataloader(self.train_data, shuffle=True)
        self.val_loader = self._create_dataloader(self.val_data, shuffle=False)
        self.test_loader = self._create_dataloader(self.test_data, shuffle=False)

        # Loss function: triplet loss based on cosine similarity
        self.criterion = self._triplet_loss

        # Set model to train mode
        self.model.model.train()

        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def _create_dataloader(self, dataset: List[Dict], shuffle: bool = True) -> DataLoader:
        """
        Creates DataLoader that yields prompt, positive image, negative image triplets with embeddings.

        Args:
            dataset (list): List of data dicts for specified split.
            shuffle (bool): Whether to shuffle data.

        Returns:
            DataLoader: with customized dataset handler.
        """
        # Create custom Dataset for batching
        custom_dataset = TripletDataset(dataset, self.model)
        return DataLoader(custom_dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=custom_dataset.collate)

    def _triplet_loss(self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss:
        max(0, ||a - p||^2 - ||a - n||^2 + margin)
        where || . || represents cosine similarity (in embedding space, vector differences).

        Args:
            a (Tensor): prompt embeddings batch
            p (Tensor): positive image embeddings batch
            n (Tensor): negative image embeddings batch

        Returns:
            Tensor: scalar loss for batch
        """
        # Cosine similarity (embeddings are normalized)
        dist_ap = torch.nn.functional.cosine_similarity(a, p)
        dist_an = torch.nn.functional.cosine_similarity(a, n)
        loss = torch.relu(dist_ap - dist_an + self.margin)
        return loss.mean()

    def train(self):
        """
        Execute training over the specified number of epochs, including validation.
        """
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            total_batches = len(self.train_loader)
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
            for batch in pbar:
                prompt_embeddings = batch['prompt_emb'].to(self.device)  # (batch_size, embed_dim)
                pos_image_embeddings = batch['pos_image_emb'].to(self.device)
                neg_image_embeddings = batch['neg_image_emb'].to(self.device)

                # Forward pass: compute triplet loss
                loss = self._triplet_loss(prompt_embeddings, pos_image_embeddings, neg_image_embeddings)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / total_batches
            logging.info(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")

            # Validate after each epoch
            val_accuracy = self.validate()
            # Save the best model based on validation accuracy
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.save_model(os.path.join(self.config['model'].get('save_dir', 'models/checkpoints/'), 'best_model.pt'))
                logging.info(f"New best model saved at epoch {epoch} with validation accuracy {val_accuracy:.2%}")

    def validate(self) -> float:
        """
        Evaluate the model over validation dataset and compute pairwise accuracy.

        Returns:
            float: validation pairwise accuracy
        """
        total_pairs = 0
        correct_pairs = 0

        try:
            self.model.model.eval()
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc='Validating'):
                    prompt_embeddings = batch['prompt_emb'].to(self.device)
                    pos_image_embeddings = batch['pos_image_emb'].to(self.device)
                    neg_image_embeddings = batch['neg_image_emb'].to(self.device)

                    # Compute scores via cosine similarity
                    scores_pos = torch.nn.functional.cosine_similarity(prompt_embeddings, pos_image_embeddings)
                    scores_neg = torch.nn.functional.cosine_similarity(prompt_embeddings, neg_image_embeddings)

                    correct_pairs += torch.sum(scores_pos > scores_neg).item()
                    total_pairs += prompt_embeddings.size(0)
        finally:
            self.model.model.train()

        accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0.0
        logging.info(f"Validation Pairwise Accuracy: {accuracy:.2%}")
        return accuracy

    def test(self):
        """
        Evaluate the model on the test dataset; output pairwise accuracy.
        """
        test_accuracy = self.validate()
        print(f"Test Pairwise Accuracy: {test_accuracy:.2%}")
        return test_accuracy

    def save_model(self, path: str):
        """
        Save the model's weights.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_weights(path)
        logging.info(f"Model weights saved at {path}")

    def load_model(self, path: str):
        """
        Load model weights from checkpoint.
        """
        self.model.load_weights(path)
        logging.info(f"Model weights loaded from {path}")

    def visualize_prompt_clustering(self):
        """
        Conducts t-SNE on prompt embeddings and plots the clusters for validation prompts.
        """
        prompts = list(self.model.prompt_embeddings_cache.keys())
        embeddings = torch.stack([self.model.prompt_embeddings_cache[p] for p in prompts])
        embeddings_np = embeddings.cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        cluster_2d = tsne.fit_transform(embeddings_np)

        plt.figure(figsize=(10, 8))
        plt.scatter(cluster_2d[:, 0], cluster_2d[:, 1], s=10, cmap='tab20')
        plt.title("t-SNE of Prompt Embeddings")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()

    def visualize_ranking(self, prompt: str, top_k: int = 5):
        """
        Ranks images associated with a prompt by Social Reward score.

        Args:
            prompt (str): The prompt to visualize rankings for.
            top_k (int): How many top images to display.
        """
        # Fetch associated images for prompt (from dataset or inference)
        # For simplicity, assume a method to get all images for prompt exists
        images_for_prompt = self._get_images_for_prompt(prompt)

        # Score images
        scores = []
        prompt_emb = self.model.encode_prompt(prompt)
        for img_path in images_for_prompt:
            img_emb = self.model.encode_image(img_path)
            score = self.model.compute_score(prompt_emb, img_emb)
            scores.append((img_path, score))
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Plot top_k images
        fig, axs = plt.subplots(1, top_k, figsize=(15, 3))
        for i in range(min(top_k, len(scores))):
            img_path, score = scores[i]
            # Load image for display
            from PIL import Image
            img = Image.open(img_path)
            axs[i].imshow(img)
            axs[i].set_title(f'Score: {score:.2f}')
            axs[i].axis('off')
        plt.suptitle(f"Top {top_k} images ranked by Social Reward for prompt:\n'{prompt}'")
        plt.show()

    def _get_images_for_prompt(self, prompt: str) -> list:
        """
        Placeholder: return the list of image paths associated with a prompt.
        In practice, this should extract images from dataset based on prompt.

        Args:
            prompt (str): prompt string

        Returns:
            list: Image paths for the prompt
        """
        # For actual implementation, link dataset loader to fetch images
        # Here, we assume dataset loader provides such method
        images_list = []
        for triplet in self.train_data + self.val_data + self.test_data:
            if triplet['prompt'] == prompt:
                images_list.append(triplet['pos_image_path'])
        # Remove duplicates
        return list(set(images_list))


class TripletDataset:
    """
    Custom Dataset for batching prompt, positive, negative image embeddings.
    It encodes prompts and images on-the-fly using the model.
    """

    def __init__(self, data: List[Dict], model: Model):
        """
        Args:
            data (list): List of triplet dicts with prompt, pos_image_path, neg_image_path.
            model (Model): To encode prompts and images.
        """
        self.data = data
        self.model = model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        triplet = self.data[index]
        prompt = triplet['prompt']
        pos_image_path = triplet['pos_image_path']
        neg_image_path = triplet['neg_image_path']

        # Encode prompt
        prompt_emb = self.model.encode_prompt(prompt)
        # Encode target images
        pos_image_emb = self.model.encode_image(pos_image_path)
        neg_image_emb = self.model.encode_image(neg_image_path)

        return {
            'prompt_emb': prompt_emb,
            'pos_image_emb': pos_image_emb,
            'neg_image_emb': neg_image_emb
        }

    def collate(self, batch):
        """
        Collate function to combine batch items into batch tensors.
        """
        prompt_embs = torch.stack([item['prompt_emb'] for item in batch], dim=0)
        pos_image_embs = torch.stack([item['pos_image_emb'] for item in batch], dim=0)
        neg_image_embs = torch.stack([item['neg_image_emb'] for item in batch], dim=0)
        return {
            'prompt_emb': prompt_embs,
            'pos_image_emb': pos_image_embs,
            'neg_image_emb': neg_image_embs
        }
