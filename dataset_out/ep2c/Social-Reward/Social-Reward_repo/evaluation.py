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

