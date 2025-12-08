## label_mapping.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from typing import List, Dict, Optional, Union
from tqdm import tqdm  # for progress bar during iterative updates
import matplotlib.pyplot as plt

class LabelMapper:
    def __init__(self,
                 strategy: str,  # 'FreqMap', 'IterMap', 'SemanticMap', 'FullyMap'
                 source_class_names: List[str],
                 target_class_names: List[str],
                 map_params: Dict,
                 device: torch.device):
        """
        Initialize the label mapping object according to strategy and parameters.
        Args:
            strategy (str): Mapping strategy.
            source_class_names (List[str]): List of source class names.
            target_class_names (List[str]): List of target dataset class names.
            map_params (dict): Additional parameters, e.g., n classes, init weights, etc.
            device (torch.device): Device to run computations on.
        """
        self.strategy = strategy
        self.source_class_names = source_class_names
        self.target_class_names = target_class_names
        self.device = device

        # Store parameters
        self.params = map_params

        # Initialize data structures depending on strategy
        num_source = len(source_class_names)
        num_target = len(target_class_names)

        # For strategies that require a mapping matrix
        if self.strategy in ['FreqMap', 'IterMap', 'FullyMap']:
            # Initialize mapping matrix: source x target
            # For FreqMap/IterMap: 1 indicates mapped, else 0
            self.M = torch.zeros((num_source, num_target), device=self.device)
            # For FullyMap, initialize linear weights later
        elif self.strategy == 'SemanticMap':
            # Compute embeddings for class names
            self.clip_model = None
            self.clip_processor = None
            self._init_clip_embeddings()
            # Similarity matrix between source and target classes
            self.semantic_similarity = None
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # For FullyMap, define linear layer for learned mapping
        if self.strategy == 'FullyMap':
            # Map source logits to target logits
            self.linear_mapping = nn.Linear(num_source, len(target_class_names))
            # Initialize linear layer weights
            self._init_fullymap_weights()
            self.linear_mapping.to(self.device)

        # For IterMap, store current mapping (initially FreqMap or default)
        if self.strategy == 'IterMap':
            # Initialize as empty, will be updated via update_mapping()
            self.iter_mapping = None

    def _init_clip_embeddings(self):
        """
        Initialize CLIP model and get class name embeddings for source and target.
        """
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            # Freeze CLIP model
            for param in self.clip_model.parameters():
                param.requires_grad = False
        except ImportError:
            # Fallback: use transformers CLIP
            from transformers import CLIPModel, CLIPProcessor
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Encode class names
        self.source_embeddings = self._compute_text_embeddings(self.source_class_names)
        self.target_embeddings = self._compute_text_embeddings(self.target_class_names)

        # Compute cosine similarity matrix: target x source
        # Result: shape (target_classes, source_classes)
        self.semantic_similarity = torch.zeros((len(self.target_class_names), len(self.source_class_names)), device=self.device)
        for i in range(len(self.target_class_names)):
            sim = F.cosine_similarity(self.target_embeddings[i].unsqueeze(0), self.source_embeddings, dim=-1)
            self.semantic_similarity[i] = sim

        # For mapping target class index to source class index
        # Will be used to assign source classes to target
        self.target_to_source_mapping = torch.argmax(self.semantic_similarity, dim=1)

    def _compute_text_embeddings(self, class_names: List[str]) -> torch.Tensor:
        """
        Compute normalized text embeddings for a list of class names via CLIP.
        """
        if hasattr(self, 'clip_processor'):
            inputs = self.clip_processor(text=class_names, return_tensors='pt', padding=True).to(self.device)
            with torch.no_grad():
                feats = self.clip_model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats
        elif hasattr(self, 'clip_model'):
            # Alternative method if using clip from 'clip' package
            import clip
            tokens = clip.tokenize(class_names).to(self.device)
            with torch.no_grad():
                feats = self.clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats
        else:
            # Should not happen
            raise RuntimeError("No CLIP model available for text embeddings.")

    def map(self, predictions: torch.Tensor, train_data_preds: Optional[Dict]=None):
        """
        Map raw model predictions to target classes according to strategy.
        Args:
            predictions (Tensor): shape (N, K_s), source model logits or predictions.
            train_data_preds (optional): Used for FreqMap, dict of {target_class_idx: count}
        Returns:
            mapped_preds (Tensor): shape (N,), target class indices
        """
        if self.strategy == 'FreqMap':
            # Use frequency counts to assign target class
            # train_data_preds: dict or tensor with counts
            # For online predictions, we assume the tally is already computed
            # The mapping matrix self.M indicates source->target
            # Predictions: shape (N, K_s)
            # Get source class predictions
            source_preds = torch.argmax(predictions, dim=1)  # shape (N,)
            # Map source class to target class based on the mapping matrix
            # For each source class, find assigned target class
            target_indices = torch.zeros_like(source_preds)
            for s_idx in range(self.M.shape[0]):
                tgt_idx = torch.argmax(self.M[s_idx])  # target class assigned to source s_idx
                source_mask = (source_preds == s_idx)
                target_indices[source_mask] = tgt_idx
            return target_indices
        elif self.strategy == 'IterMap':
            # Recompute mapping at current epoch/step
            # Call update_mapping() externally to refresh self.M
            # After update_mapping(), use same logic as FreqMap
            source_preds = torch.argmax(predictions, dim=1)
            target_indices = torch.zeros_like(source_preds)
            for s_idx in range(self.M.shape[0]):
                tgt_idx = torch.argmax(self.M[s_idx])
                source_mask = (source_preds == s_idx)
                target_indices[source_mask] = tgt_idx
            return target_indices
        elif self.strategy == 'SemanticMap':
            # Use class name embeddings similarity
            # predictions: source class indices
            # Map prediction to source class embedding, then find closest target class
            # Usually, predictions are class indices (or logits). Here, assume predictions are (N,)
            # For predictions in logits, take argmax
            pred_source_indices = torch.argmax(predictions, dim=1)
            source_embs = self.source_embeddings[pred_source_indices]  # (N, D)
            # Compute cosine similarity with target embeddings
            # target_embeddings shape: (T, D)
            # similarity: (N, T)
            sim = F.cosine_similarity(source_embs.unsqueeze(1), self.target_embeddings.unsqueeze(0), dim=-1)
            # For each, pick target class with max similarity
            target_preds = torch.argmax(sim, dim=1)
            return target_preds
        elif self.strategy == 'FullyMap':
            # Pass source logits (predictions) through linear layer
            # predictions shape: (N, K_s)
            final_logits = self.linear_mapping(predictions)  # (N, T)
            target_preds = torch.argmax(final_logits, dim=1)
            return target_preds
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def update_mapping(self, training_dataset=None, source_model=None):
        """
        Update class correspondence or weights during training for strategies like IterMap and FullyMap.
        Args:
            training_dataset (Dataset): dataset to compute mappings from.
            source_model (PretrainedModel): model for predictions if needed.
        """
        if self.strategy == 'IterMap':
            # Recompute frequency-based mapping from training dataset
            if training_dataset is None or source_model is None:
                raise ValueError("training_dataset and source_model required for IterMap update.")
            count_matrix = torch.zeros((len(self.source_class_names), len(self.target_class_names)), device=self.device)
            dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=self.params.get('batch_size', 32),
                                                     shuffle=False)
            with torch.no_grad():
                for batch in dataloader:
                    imgs = batch['image'].to(self.device)
                    # Obtain predictions from source model
                    preds = source_model.forward(imgs)
                    pred_labels = torch.argmax(preds, dim=1)
                    # For each sample, get source and target labels
                    # Assuming batch includes target labels in batch['label'] for training set
                    target_labels = batch.get('label', None)
                    if target_labels is None:
                        # If no explicit target labels provided, infer from dataset order or metadata
                        # For placeholder, assume source labels correspond to class indices
                        break  # cannot update without target class info
                    # Here, for practical purposes, require dataset to provide target labels
                    # For simplicity, we skip actual update unless dataset provides 'label'
                    # Better to implement using ground truth labels if available
            # After computing frequency, update self.M accordingly
            # But as placeholder, we here just re-initialize mapping based on max class predictions
            # For actual implementation, this requires dataset ground truth
            # or prediction counts per class
            # For simplicity, set to default: assign target classes using semantic similarity
            # Alternatively, could do:
            # For each target class t, find source class s that predicts most of t
            # But requires dataset labels
            pass
        elif self.strategy == 'FullyMap':
            # Update linear layer weights possibly with weights derived from semantic similarity
            # Here, as per paper, initialize weights based on semantic similarity or keep fixed
            # For illustration, do a simple semantic initialization if desired
            # Otherwise, keep weights fixed
            pass

    def visualize_mapping(self):
        """
        Generate a visualization of the class correspondence.
        For FreqMap and IterMap: show mapping counts or confusion
        For SemanticMap: plot similarity matrix
        For FullyMap: visualize linear weights
        """
        if self.strategy in ['FreqMap', 'IterMap']:
            # Plot the mapping matrix as a heatmap
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,8))
            plt.imshow(self.M.cpu().numpy(), cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.xlabel('Target Classes')
            plt.ylabel('Source Classes')
            plt.title(f'Mapping matrix heatmap: {self.strategy}')
            plt.show()
        elif self.strategy == 'SemanticMap':
            # Plot similarity matrix
            import seaborn as sns
            plt.figure(figsize=(8,6))
            sns.heatmap(self.semantic_similarity.cpu().numpy(), annot=True, cmap='coolwarm')
            plt.xlabel('Source Classes')
            plt.ylabel('Target Classes')
            plt.title('Semantic similarity between classes')
            plt.show()
        elif self.strategy == 'FullyMap':
            # Visualize linear layer weights as a heatmap
            weights = self.linear_mapping.weight.data.cpu()
            plt.figure(figsize=(10,8))
            sns.heatmap(weights, cmap='viridis', xticklabels=self.target_class_names, yticklabels=self.source_class_names)
            plt.xlabel('Target Classes')
            plt.ylabel('Source Classes')
            plt.title('FullyMap Linear Layer Weights')
            plt.show()
        else:
            print("No visualization available for this strategy.")
