## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
from dataset_loader import DatasetLoader
from model import SpikingGraphReasoningModel
from utils import surrogate_gradient, delay_quantize
from evaluation import Evaluator

class Trainer:
    """
    Manages training and evaluation of the GRSNN model for graph reasoning tasks.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer with configuration, datasets, model, optimizer, and evaluation metrics.
        """
        self.config = config
        self.device = torch.device(config['misc'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        torch.manual_seed(config['misc'].get('seed', 42))
        np.random.seed(config['misc'].get('seed', 42))
        
        # Load dataset
        self.data_loader = DatasetLoader(
            dataset_path=config['dataset']['path'],
            dataset_type=config['dataset'].get('type', 'knowledge_graph'),
            batch_size=config['dataset'].get('batch_size', 32),
            negative_samples=config['dataset'].get('negative_samples', 50),
            seed=config['misc'].get('seed', 42)
        )
        dataset = self.data_loader.load_data()
        self.train_triplets = dataset['train']['triplets'].to(self.device)
        self.val_triplets = dataset['val']['triplets'].to(self.device)
        self.test_triplets = dataset['test']['triplets'].to(self.device)
        self.num_entities = self.data_loader.num_entities
        self.num_relations = self.data_loader.num_relations

        # Load relation and entity embeddings
        self.entity_embeddings = nn.Embedding(self.num_entities, 32).to(self.device)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(self.num_relations, 64).to(self.device)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # Initialize the model
        self.model = SpikingGraphReasoningModel(
            config=self.config,
            num_entities=self.num_entities,
            num_relations=self.num_relations
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            list(self.model.parameters()) +
            list(self.entity_embeddings.parameters()) +
            list(self.relation_embeddings.parameters()),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 1e-5)
        )

        # Loss criterion
        self.criterion = nn.BCEWithLogitsLoss()

        # Evaluation
        self.evaluator = Evaluator(
            model=self.model,
            entity_embeddings=self.entity_embeddings,
            relation_embeddings=self.relation_embeddings,
            data_loader=self.data_loader,
            config=self.config,
            device=self.device
        )

        # Save configs for later
        self.epoch = 0
        self.best_mrr = 0
        self.best_model_state = None
        self.early_stop_counter = 0

    def train(self):
        """
        Main training loop over epochs.
        """
        max_epochs = self.config['training'].get('epochs', 20)
        patience = self.config['training'].get('early_stopping_patience', 5)
        grad_clip = self.config['training'].get('gradient_clip', 0.5)

        for epoch in range(1, max_epochs + 1):
            self.epoch = epoch
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            # Batch loader function iterable
            for batch_idx, batch_data in enumerate(self.data_loader.get_batch(self.train_triplets)):
                # batch_data: dict with 'pos_triplets', 'neg_triplets'
                pos_triplets = batch_data['pos_triplets'].to(self.device)
                neg_triplets = batch_data['neg_triplets'].to(self.device)

                # Clear gradients
                self.optimizer.zero_grad()

                # Forward pass for positive triplets
                pos_scores, pos_spikes = self._forward_triplet_batch(pos_triplets)
                # Decode for positive triplets
                pos_decoded = self.model.decode_spike_trains(pos_spikes)

                # Forward for negative triplets
                neg_scores, neg_spikes = self._forward_triplet_batch(neg_triplets)
                neg_decoded = self.model.decode_spike_trains(neg_spikes)

                # Compute losses
                pos_labels = torch.ones(pos_scores.shape[0], 1, device=self.device)
                neg_labels = torch.zeros(neg_scores.shape[0], 1, device=self.device)
                score_pred = torch.cat([pos_scores, neg_scores], dim=0)
                labels = torch.cat([pos_labels, neg_labels], dim=0)

                # Using BCEWithLogitsLoss on the raw scores, e.g., from predictor g
                loss = self.criterion(score_pred.squeeze(), labels.squeeze())

                # Backpropagate surrogate gradients
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(self.entity_embeddings.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(self.relation_embeddings.parameters(), grad_clip)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")

            # Validation and early stopping
            val_metrics = self.evaluator.evaluate(self.val_triplets)
            current_mrr = val_metrics.get('MRR', 0)
            print(f"Validation MRR: {current_mrr:.4f}")
            if current_mrr > self.best_mrr:
                self.best_mrr = current_mrr
                self.best_model_state = {
                    'model_state_dict': self.model.state_dict(),
                    'entity_embeddings': self.entity_embeddings.state_dict(),
                    'relation_embeddings': self.relation_embeddings.state_dict()
                }
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
            self.entity_embeddings.load_state_dict(self.best_model_state['entity_embeddings'])
            self.relation_embeddings.load_state_dict(self.best_model_state['relation_embeddings'])
            print("Loaded best model based on validation performance.")

    def _forward_triplet_batch(self, triplets: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """
        Forward propagate a batch of triplets through the SNN.
        Args:
            triplets (Tensor): shape (batch_size, 3)
        Returns:
            scores (Tensor): likelihood scores per triplet
            spike_trains (list): list of spike train tensors per batch
        """
        batch_size = triplets.shape[0]
        # Obtain embeddings for entities and relations
        entity_embs = self.entity_embeddings
        rel_embs = self.relation_embeddings
        adjacency_list = self.data_loader.adj_dict

        # Run model's propagate function
        pair_reps, spike_trains = self.model(
            batch_triplets=triplets,
            entity_embeddings=entity_embs,
            relation_embeddings=rel_embs,
            adjacency_list=adjacency_list,
            device=self.device
        )  # pair_reps: (batch_size, neuron_count), spike_trains: list of tensors

        # Compute likelihood scores via predictor network
        scores = self.model.predictor(pair_reps).squeeze(1)
        return scores, spike_trains

    def evaluate(self, triplets: torch.Tensor):
        """
        Evaluate model on given triplets, returning metrics.
        """
        self.model.eval()
        with torch.no_grad():
            scores, spike_trains = self._forward_triplet_batch(triplets)
            # Decode spike trains for all triplets
            decoded = self.model.decode_spike_trains(spike_trains)
            # Compute evaluation metrics using Evaluator
            metrics = self.evaluator.compute_metrics(decoded, triplets)
        return metrics

    def run(self):
        """
        Run full training and evaluation, then test.
        """
        self.train()
        print("Training complete. Evaluating on test set.")
        test_metrics = self.evaluate(self.test_triplets)
        print(f"Test Metrics: {test_metrics}")
        # Save final model if needed
        torch.save(self.model.state_dict(), 'final_grsnn_model.pth')

