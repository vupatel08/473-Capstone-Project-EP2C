# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py

import os
import json
import random
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer

from tqdm import tqdm

class DatasetLoader:
    """
    DatasetLoader is responsible for:
      - Loading triplet data (prompt, positive, negative images, and signals)
      - Filtering based on view counts, NSFW flags, prompt length
      - Clustering prompts using hierarchical clustering
      - Splitting into train, val, test sets ensuring prompt-level separation
    """

    def __init__(self,
                 dataset_path: str,
                 max_prompt_words: int = 5,
                 min_view_count: int = 10,
                 max_view_count: int = 1000,
                 filter_negative_signals: bool = True,
                 seed: int = 42,
                 clustering_method: str = "ward",
                 num_clusters: int = 173,
                 prompt_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 verbose: bool = True):
        """
        Initializes the DatasetLoader with dataset path and preprocessing parameters.

        Args:
            dataset_path (str): Path to the curated dataset directory/csv file.
            max_prompt_words (int): Max number of words in prompt.
            min_view_count (int): Minimum view count for images.
            max_view_count (int): Maximum view count for images.
            filter_negative_signals (bool): Whether to filter images with zero remix counts.
            seed (int): Random seed for reproducibility.
            clustering_method (str): Linkage method for hierarchical clustering.
            num_clusters (int): Number of clusters for prompt clustering.
            prompt_embedding_model_name (str): Model name for prompt embedding.
            verbose (bool): Whether to display progress info.
        """
        self.dataset_path = dataset_path
        self.max_prompt_words = max_prompt_words
        self.min_view_count = min_view_count
        self.max_view_count = max_view_count
        self.filter_negative_signals = filter_negative_signals
        self.seed = seed
        self.clustering_method = clustering_method
        self.num_clusters = num_clusters
        self.prompt_embedding_model_name = prompt_embedding_model_name
        self.verbose = verbose

        # Containers for data
        self.raw_data = []  # will hold raw triplet dicts
        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.prompt_embeddings_cache = {}  # prompt text -> embedding tensor
        self.prompt_cluster_labels = {}  # prompt text -> cluster id

        # Load and process dataset
        self._load_data()
        self._filter_data()
        self._compute_prompt_embeddings()
        self._perform_clustering()
        self._split_data()

    def _load_data(self):
        """
        Loads the raw dataset from files.
        Assumes data is stored in a CSV or JSONL with necessary fields:
        prompt, pos_image_path, neg_image_path, remix_counts, creator_signal, view_counts, nsfw_flag
        """
        # Infer file format from dataset_path: assume CSV for simplicity
        try:
            data_df = pd.read_csv(self.dataset_path)
        except Exception:
            # fallback: try JSONL
            with open(self.dataset_path, 'r') as f:
                lines = f.readlines()
            data_list = [json.loads(line) for line in lines]
            data_df = pd.DataFrame(data_list)

        # Expected columns:
        # 'prompt', 'pos_image_path', 'neg_image_path',
        # 'remix_counts', 'creator_signal', 'view_counts', 'nsfw_flag'
        required_cols = ['prompt', 'pos_image_path', 'neg_image_path',
                         'remix_counts', 'creator_signal', 'view_counts', 'nsfw_flag']
        for col in required_cols:
            if col not in data_df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert to list of dicts for flexibility
        for _, row in data_df.iterrows():
            triplet = {
                'prompt': row['prompt'],
                'pos_image_path': row['pos_image_path'],
                'neg_image_path': row['neg_image_path'],
                'remix_counts': float(row['remix_counts']),
                'creator_signal': bool(row['creator_signal']),
                'view_counts': float(row['view_counts']),
                'nsfw_flag': bool(row['nsfw_flag'])
            }
            self.raw_data.append(triplet)

        if self.verbose:
            print(f"Loaded {len(self.raw_data)} total triplets from dataset.")

    def _filter_data(self):
        """
        Applies filtering based on:
        - View counts (min and max)
        - NSFW flag
        - Prompt length
        - Remix/signals if specified
        """
        filtered_data = []

        for triplet in self.raw_data:
            # Filter NSFW
            if triplet['nsfw_flag']:
                continue

            # Filter view counts
            view_count = triplet['view_counts']
            if view_count < self.min_view_count or view_count > self.max_view_count:
                continue

            # Filter prompt length
            prompt_word_count = len(triplet['prompt'].split())
            if prompt_word_count > self.max_prompt_words:
                continue

            # Filter remix signals if applicable
            if self.filter_negative_signals:
                # Keep only images with remix_counts >= 1 for positives
                # For negatives, remix_counts should be zero
                # But we will handle in sampling phase; for dataset, keep all for now
                pass

            filtered_data.append(triplet)

        self.filtered_data = filtered_data
        if self.verbose:
            print(f"After filtering, {len(self.filtered_data)} triplets remain.")

    def _compute_prompt_embeddings(self):
        """
        Computes embeddings for all prompts using SentenceTransformer.
        Caches embeddings for efficiency.
        """
        self.model = SentenceTransformer(self.prompt_embedding_model_name)
        prompts = list(set([triplet['prompt'] for triplet in self.filtered_data]))
        self.prompt_embeddings_cache = {}

        if self.verbose:
            print(f"Computing embeddings for {len(prompts)} prompts...")

        for prompt in tqdm(prompts, desc="Encoding prompts"):
            emb = self.model.encode(prompt, convert_to_tensor=True)
            self.prompt_embeddings_cache[prompt] = emb

    def _perform_clustering(self):
        """
        Performs hierarchical clustering on prompt embeddings to assign cluster labels.
        """
        # Collect unique prompts and their embeddings
        prompts = list(self.prompt_embeddings_cache.keys())
        embeddings = torch.stack([self.prompt_embeddings_cache[prompt] for prompt in prompts])
        embeddings_np = embeddings.cpu().numpy()

        # Compute pairwise distances
        if self.verbose:
            print("Performing hierarchical clustering...")
        Z = linkage(pdist(embeddings_np, metric='cosine'), method=self.clustering_method)

        # Assign cluster labels with a fixed number of clusters
        cluster_labels = fcluster(Z, t=self.num_clusters, criterion='maxclust')

        # Map prompts to cluster ids
        for prompt, label in zip(prompts, cluster_labels):
            self.prompt_cluster_labels[prompt] = label

        if self.verbose:
            cluster_counts = np.bincount(cluster_labels)
            print(f"Clustering complete. Cluster counts: {cluster_counts}")

    def _split_data(self):
        """
        Splits data into train, val, test sets based on prompt clusters or prompt texts.
        Ensure no prompt appears in multiple sets.
        """
        # Collect all unique prompts
        prompts = list(set([triplet['prompt'] for triplet in self.filtered_data]))
        random.seed(self.seed)

        # Shuffle prompts
        random.shuffle(prompts)

        total = len(prompts)
        train_cut = int(0.7 * total)
        val_cut = int(0.8 * total)

        # Assign prompts to splits
        train_prompts = set(prompts[:train_cut])
        val_prompts = set(prompts[train_cut:val_cut])
        test_prompts = set(prompts[val_cut:])

        # Allocate triplets based on their prompt
        for triplet in self.filtered_data:
            prompt = triplet['prompt']
            if prompt in train_prompts:
                self.train_data.append(triplet)
            elif prompt in val_prompts:
                self.val_data.append(triplet)
            else:
                self.test_data.append(triplet)

        if self.verbose:
            print(f"Data split: {len(self.train_data)} train, {len(self.val_data)} val, {len(self.test_data)} test triplets.")

    def get_train_data(self) -> List[Dict]:
        """Return list of training triplets."""
        return self.train_data

    def get_val_data(self) -> List[Dict]:
        """Return list of validation triplets."""
        return self.val_data

    def get_test_data(self) -> List[Dict]:
        """Return list of test triplets."""
        return self.test_data

    def get_prompt_embedding(self, prompt: str) -> torch.Tensor:
        """Return the embedding tensor for a specific prompt."""
        return self.prompt_embeddings_cache.get(prompt, None)

    def get_prompt_cluster_id(self, prompt: str) -> int:
        """Return the cluster ID of a prompt."""
        return self.prompt_cluster_labels.get(prompt, -1)

    def get_triplet_sample(self, split: str = 'train') -> Tuple[str, str, str]:
        """
        Sample a triplet from the specified split.

        Args:
            split (str): 'train', 'val', or 'test'

        Returns:
            Tuple[prompt, positive_image_path, negative_image_path]
        """
        data_split = {
            'train': self.train_data,
            'val': self.val_data,
            'test': self.test_data
        }
        data_list = data_split.get(split, [])
        if not data_list:
            raise ValueError(f"No data available for split: {split}")
        triplet = random.choice(data_list)
        return (triplet['prompt'], triplet['pos_image_path'], triplet['neg_image_path'])
```

## evaluation.py

```python
## evaluation.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# Import the Model class from model.py
from model import Model

class Evaluation:
    """
    The Evaluation class assesses the trained Social Reward model's performance on a test dataset,
    visualizes rankings, prompt embeddings with t-SNE, and performs per-cluster accuracy analysis.
    """

    def __init__(
        self,
        model_checkpoint_path: str,
        dataset_test: List[Dict],
        dataset_name: str = "TestSet",
        config: Optional[Dict] = None,
        device: Optional[str] = None
    ):
        """
        Loads the trained model, sets up device, and prepares datasets.
        Args:
            model_checkpoint_path (str): Path to the trained model weights.
            dataset_test (list): List of triplets for testing.
            dataset_name (str): Name for dataset, e.g., "Test".
            config (dict, optional): Configuration parameters.
            device (str, optional): 'cuda' or 'cpu'. Defaults to if provided, else auto.
        """
        # Load device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Instantiate the model
        pretrained_model_name = "openai/clip-vit-base-patch32"
        self.model = Model(pretrained_model_name=pretrained_model_name, load_weights=model_checkpoint_path, device=self.device)
        self.model.model.eval()
        self.model.model.to(self.device)

        # Save dataset for analysis
        self.dataset_test = dataset_test
        self.dataset_name = dataset_name

        # Prepare embedding cache if needed
        self.prompt_embeddings_cache = {}
        # Load the prompt transformer for embeddings (sentence-transformers)
        self.prompt_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # Build a prompt list from data for embedding
        self.prompt_list = list({triplet['prompt'] for triplet in self.dataset_test})

        # Compute prompt embeddings once for clustering and analysis
        print("Encoding prompts for t-SNE visualization...")
        self.prompt_embeddings = {}
        for prompt in tqdm(self.prompt_list, desc="Prompt embedding"):
            emb = self.prompt_model.encode(prompt, convert_to_tensor=True)
            self.prompt_embeddings[prompt] = emb

        # For per-cluster analysis, initialize cluster labels (will be assigned externally if needed)
        self.prompt_clusters = None

        print("Evaluation setup complete.")

    def compute_pairwise_accuracy(self) -> float:
        """
        Computes the pairwise accuracy over the test dataset.
        Compares model scores on (prompt, positive image, negative image) triplets.
        Returns:
            accuracy (float): Percentage of correct preference predictions.
        """
        total_pairs = 0
        correct_pairs = 0

        with torch.no_grad():
            for triplet in tqdm(self.dataset_test, desc="Evaluating"):
                prompt = triplet['prompt']
                pos_img_path = triplet['pos_image_path']
                neg_img_path = triplet['neg_image_path']

                prompt_emb = self.model.encode_prompt(prompt)
                pos_img_emb = self.model.encode_image(pos_img_path)
                neg_img_emb = self.model.encode_image(neg_img_path)

                score_pos = self.model.compute_score(prompt_emb, pos_img_emb)
                score_neg = self.model.compute_score(prompt_emb, neg_img_emb)

                if score_pos > score_neg:
                    correct_pairs += 1
                total_pairs += 1

        accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0.0
        print(f"{self.dataset_name} Pairwise Accuracy: {accuracy:.4f} ({correct_pairs}/{total_pairs})")
        return accuracy

    def visualize_ranking(self, prompts: Optional[List[str]] = None, top_k: int = 5):
        """
        Plots images ranked by Social Reward scores for selected prompts.
        Args:
            prompts (list, optional): List of prompts to visualize. Defaults to all unique prompts.
            top_k (int): Number of top images to display per prompt.
        """
        if prompts is None:
            prompts = self.prompt_list

        for prompt in prompts:
            # Collect all images associated with the prompt
            images_for_prompt = [
                triplet['pos_image_path']
                for triplet in self.dataset_test
                if triplet['prompt'] == prompt
            ]
            if not images_for_prompt:
                continue

            # Encode prompt once
            prompt_emb = self.model.encode_prompt(prompt)

            # Compute scores for each image
            scores = []
            for img_path in images_for_prompt:
                img_emb = self.model.encode_image(img_path)
                score = self.model.compute_score(prompt_emb, img_emb)
                scores.append((img_path, score))

            # Sort images by score descending
            scores.sort(key=lambda x: x[1], reverse=True)

            # Plot top_k images
            fig, axs = plt.subplots(1, min(top_k, len(scores)), figsize=(15, 3))
            if len(scores) == 1:
                axs = [axs]
            for i, (img_path, score) in enumerate(scores[:top_k]):
                from PIL import Image
                img = Image.open(img_path)
                axs[i].imshow(img)
                axs[i].set_title(f"Score: {score:.2f}")
                axs[i].axis('off')
            plt.suptitle(f"Top {top_k} images ranked by Social Reward for prompt:\n'{prompt}'")
            plt.tight_layout()
            plt.show()

    def generate_prompt_tsne(self, perplexity: int = 30):
        """
        Performs t-SNE on prompt embeddings and plots the 2D visualization.
        Args:
            perplexity (int): Perplexity parameter for t-SNE.
        """
        # Convert prompt embeddings to numpy array
        embed_list = [emb.cpu().numpy() for emb in self.prompt_embeddings.values()]
        embed_matrix = np.stack(embed_list, axis=0)

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_results = tsne.fit_transform(embed_matrix)

        plt.figure(figsize=(10,8))
        plt.scatter(tsne_results[:,0], tsne_results[:,1], s=10, cmap='tab10')
        # Optional: annotate prompts for interpretability
        # for i, prompt in enumerate(self.prompt_list):
        #     if i % 100 == 0:
        #         plt.annotate(prompt[:10], (tsne_results[i,0], tsne_results[i,1]))
        plt.title("t-SNE of Prompt Embeddings")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()

    def evaluate_per_cluster_accuracy(self):
        """
        Computes model accuracy within each prompt cluster, if cluster labels are available.
        Assumes self.prompt_clusters is a dict: prompt -> cluster_id.
        """
        if self.prompt_clusters is None:
            print("Prompt cluster labels are not assigned. Cannot compute per-cluster accuracy.")
            return

        cluster_ids = set(self.prompt_clusters.values())
        cluster_accuracy = {}

        for cluster_id in cluster_ids:
            # Gather triplets belonging to this cluster
            triplets_in_cluster = [
                triplet for triplet in self.dataset_test
                if self.prompt_clusters.get(triplet['prompt'], -1) == cluster_id
            ]
            if not triplets_in_cluster:
                continue
            correct = 0
            total = 0
            for triplet in triplets_in_cluster:
                prompt = triplet['prompt']
                pos_img_path = triplet['pos_image_path']
                neg_img_path = triplet['neg_image_path']
                prompt_emb = self.model.encode_prompt(prompt)
                pos_emb = self.model.encode_image(pos_img_path)
                neg_emb = self.model.encode_image(neg_img_path)

                score_pos = self.model.compute_score(prompt_emb, pos_emb)
                score_neg = self.model.compute_score(prompt_emb, neg_emb)

                if score_pos > score_neg:
                    correct += 1
                total += 1
            accuracy = correct / total if total > 0 else 0.0
            cluster_accuracy[cluster_id] = accuracy

        # Plot or print per-cluster accuracy
        print("Per-cluster accuracy:")
        for cid, acc in cluster_accuracy.items():
            print(f"Cluster {cid}: {acc:.4f}")

    def save_embedding_visualization(self, filename: str = "prompt_tsne.png"):
        """
        Saves a static image of the t-SNE embedding scatter plot.
        Args:
            filename (str): Path to save the plot.
        """
        embed_list = [emb.cpu().numpy() for emb in self.prompt_embeddings.values()]
        embed_matrix = np.stack(embed_list, axis=0)

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_results = tsne.fit_transform(embed_matrix)

        plt.figure(figsize=(10,8))
        plt.scatter(tsne_results[:,0], tsne_results[:,1], s=10, cmap='tab10')
        plt.title("t-SNE of Prompt Embeddings")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.savefig(filename)
        plt.close()

    def run_full_evaluation(self):
        """
        Runs the full suite of evaluation: accuracy, ranking visualization, t-SNE, per-cluster accuracy.
        """
        print("Running pairwise accuracy evaluation...")
        self.compute_pairwise_accuracy()
        print("Visualizing ranking for sample prompts...")
        self.visualize_ranking()
        print("Performing t-SNE visualization of prompts...")
        self.generate_prompt_tsne()
        print("Computing per-cluster accuracy...")
        self.evaluate_per_cluster_accuracy()
        print("Evaluation complete.")

```

## main.py

```python
# main.py
import argparse
import os
import random
import numpy as np
import torch
import yaml

from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation
from utils import set_seed, plot_prompt_clusters, plot_ranking_visualization

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Social Reward Evaluation Framework")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "finetune"], default="train", help="Operation mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to load")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Directory to save models and logs")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device(config['training'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    
    # Set seeds for reproducibility
    seed = config['training'].get('seed', 42)
    set_seed(seed)

    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and process dataset
    dataset_path = config['dataset']['dataset_path']
    loader = DatasetLoader(
        dataset_path=dataset_path,
        max_prompt_words=config['dataset'].get('max_prompt_words',5),
        min_view_count=config['dataset'].get('min_view_count',10),
        max_view_count=config['dataset'].get('max_view_count',1000),
        filter_negative_signals=True,
        seed=seed,
        clustering_method=config.get('clustering', {}).get('method', 'ward'),
        num_clusters=config.get('clustering', {}).get('num_clusters', 173)
    )
    # Dataset splits
    train_data = loader.get_train_data()
    val_data = loader.get_val_data()
    test_data = loader.get_test_data()

    # Initialize model
    pretrained_model_name = config['model'].get('pretrained_model_name', "openai/clip-vit-base-patch32")
    model_path = args.checkpoint if args.checkpoint else None
    social_reward_model = Model(pretrained_model_name=pretrained_model_name, load_weights=model_path, device=device)

    # Depending on mode, execute different workflows
    if args.mode == "train":
        # Initialize trainer
        trainer = Trainer(
            model=social_reward_model,
            train_dataset=train_data,
            val_dataset=val_data,
            test_dataset=test_data,
            config=config
        )
        # Train the model
        trainer.train()
        # Save final model weights
        save_path = os.path.join(args.output_dir, 'social_reward_final.pt')
        trainer.save_model(save_path)

        # Optional: visualize prompt clustering after training
        trainer.visualize_prompt_clustering()

    elif args.mode == "evaluate":
        # Load the trained model weights if checkpoint provided
        if args.checkpoint:
            social_reward_model.load_weights(args.checkpoint)
        else:
            print("No checkpoint provided; evaluating with current model weights.")

        # Evaluate on test set
        evaluator = Evaluation(
            model_checkpoint_path=args.checkpoint if args.checkpoint else None,
            dataset_test=test_data,
            dataset_name="TestSet"
        )
        # Compute pairwise accuracy
        pairwise_acc = evaluator.compute_pairwise_accuracy()
        print(f"Pairwise accuracy on test set: {pairwise_acc:.4%}")

        # Visualize ranking examples
        evaluator.visualize_ranking()

        # Optional: perform t-SNE visualization of prompts
        evaluator.generate_prompt_tsne()

        # Optional per-cluster analysis
        evaluator.run_full_evaluation()

    elif args.mode == "finetune":
        # Assumed available: trained social reward model checkpoint
        if args.checkpoint is None:
            raise ValueError("Checkpoint must be provided for fine-tuning mode.")

        # Load existing weights
        social_reward_model.load_weights(args.checkpoint)

        # Placeholder for fine-tuning process:
        # 1. Generate top/bottom images for prompts
        # 2. Use Social Reward scores to fine-tune generative model (not implemented here due to scope)
        # 3. Save fine-tuned generator weights
        print("Fine-tuning mode selected, but implementation depends on generative model specifics.")
        # This could involve RLHF routines or supervised re-training, which are complex workflows.
        # For now, just a placeholder.
        pass

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

if __name__ == "__main__":
    main()
```

## model.py

```python
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
```

## trainer.py

```python
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
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\Social-Reward\Social-Reward_repo`
