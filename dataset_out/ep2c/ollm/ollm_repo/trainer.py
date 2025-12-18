## trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from typing import Dict, Tuple, List
from tqdm import tqdm

from evaluation import GraphMetrics
from graph_utils import remove_cycles
from model import BingModel
from dataset_loader import Dataset, DatasetLoader
from prompt_generator import PromptGenerator

import yaml
import os

# Load configuration from YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set random seed for reproducibility
seed = config.get('misc', {}).get('seed', 42)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class Trainer:
    def __init__(
        self,
        model: BingModel,
        dataset: Dataset,
        val_dataset: Dataset,
        config: Dict,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.model = model
        self.train_dataset = dataset
        self.val_dataset = val_dataset
        self.device = device
        self.config = config

        # Hyperparameters from config
        self.learning_rate = config['training'].get('learning_rate', 1e-5)
        self.batch_size = config['training'].get('batch_size', 16)
        self.epochs = config['training'].get('epochs', 2)
        self.loss_masking = config['training'].get('loss_masking', True)
        self.relation_masking_M = config['training'].get('relation_masking_M', 100)
        self.mask_mask_prob = config['training'].get('mask_mask_prob', 0.5)
        self.gradient_clipping_norm = config['training'].get('gradient_clipping_norm', 1.0)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=self.learning_rate)

        # For validation metrics
        self.val_metrics = GraphMetrics()
        self.best_val_score = -float('inf')
        self.best_model_path = "best_model.pt"

        # Prepare dataloader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        # Batch is a list of tuples: (prompt, target_sequence, concepts)
        prompts = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        # Tokenize inputs
        inputs = self.model.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
        labels = self.model.tokenizer(targets, padding=True, truncation=True, return_tensors='pt')

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        label_ids = labels['input_ids']

        # Create mask flags and relation counts placeholders
        # These should be constructed based on parsing target sequences
        mask_flags = torch.zeros_like(label_ids, dtype=torch.bool)
        relation_counts = {}  # For the masked loss; placeholder here
        return input_ids, attention_mask, label_ids, mask_flags, relation_counts

    def _apply_loss_mask(self, loss, mask_flags, relation_counts):
        # Placeholder: implement real masking logic based on relation frequency
        # For simplicity, assume no masking; in practice, mask tokens at relation positions
        return loss

    def train(self):
        num_training_steps = len(self.train_loader) * self.epochs
        progress_bar = tqdm(range(num_training_steps), desc='Training')
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            self.model.model.train()
            for batch_idx, batch in enumerate(self.train_loader):
                input_ids, attention_mask, label_ids, mask_flags, relation_counts = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                label_ids = label_ids.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=label_ids
                )
                loss = outputs.loss
                # Apply custom masked loss if enabled
                if self.loss_masking:
                    loss = self._apply_loss_mask(loss, mask_flags, relation_counts)
                loss.backward()

                # Gradient clipping
                if self.gradient_clipping_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), self.gradient_clipping_norm)

                self.optimizer.step()
                epoch_loss += loss.item()
                progress_bar.update(1)

            # Validation after each epoch
            val_score = self.evaluate()
            print(f"Epoch {epoch} validation {self.config['evaluation']['validation_metric']}: {val_score:.4f}")

            # Save best model based on validation metric
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                torch.save(self.model.model.state_dict(), self.best_model_path)
                print(f"New best model saved at epoch {epoch}")

    def evaluate(self) -> float:
        """Evaluate model on validation set, returning main metric (e.g., Graph F1)."""
        self.model.model.eval()
        all_generated_graphs = []
        all_true_graphs = []

        for batch in self.val_loader:
            input_ids, attention_mask, label_ids, mask_flags, relation_counts = batch
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # Generate output sequences
            with torch.no_grad():
                generated_ids = self.model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9
                )
            generated_texts = [self.model.tokenizer.decode(gid, skip_special_tokens=True) for gid in generated_ids]

            # Parse generated texts into graphs
            generated_graphs = [self._parse_generated_text(text) for text in generated_texts]
            true_graphs = []  # For validation, true graphs should be preprocessed/available
            # Here, we should load or have access to the true ground truth graphs for validation
            # Assume function: get_ground_truth_graphs(batch_indices)
            # For simplicity, skipping actual ground truth loading

            all_generated_graphs.extend(generated_graphs)
            # all_true_graphs.extend(true_graphs)

        # Compute the main validation metric, e.g., Graph F1
        # At this stage, placeholders:
        main_metric = 0.0
        # For actual implementation, compare all_generated_graphs to all_true_graphs
        # e.g.,
        # main_metric = self.val_metrics.graph_f1_score(pred_graph, true_graph)

        return main_metric

    def run(self):
        """Main entry point to start training and evaluation."""
        self.train()

    def _parse_generated_text(self, text: str):
        """Parse the generated text output into a graph structure."""
        # Implement regex or parser based on the linearization schema
        # Placeholder: return empty graph
        from graph_utils import Graph
        return Graph()

# Example of usage
if __name__ == "__main__":
    # Load datasets
    loader = DatasetLoader(config)
    train_dataset = loader.load_wikipedia()
    val_dataset = loader.load_wikipedia()  # or validate on a separate split if available

    # Initialize model
    model = BingModel(
        model_name=config['model']['base_model_name'],
        use_lora=config['model'].get('use_lora', True),
        lora_rank=config['model'].get('lora_rank', 32),
        finetune_on_dataset='wikipedia',
        loss_masking=True
    )

    # Initialize trainer
    trainer = Trainer(model, train_dataset, val_dataset, config)

    # Run training routine
    trainer.run()
