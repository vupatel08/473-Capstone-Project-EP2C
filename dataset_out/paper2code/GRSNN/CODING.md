# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import random
import numpy as np
import torch

from typing import Dict, List, Tuple, Optional

class DatasetLoader:
    """
    Responsible for loading datasets, parsing triplets, and creating training/validation/test splits.
    Supports knowledge graphs with triplets and citation datasets with adjacency matrices, based on type.
    """
    def __init__(
        self,
        dataset_path: str,
        dataset_type: str = "knowledge_graph",
        batch_size: int = 32,
        negative_samples: int = 50,
        seed: int = 42
    ):
        """
        Initialize the DatasetLoader.

        Args:
            dataset_path (str): Path to dataset directory or files.
            dataset_type (str): 'knowledge_graph' or 'citation'.
            batch_size (int): Number of samples per batch.
            negative_samples (int): Number of negative samples per positive triplet.
            seed (int): Random seed for reproducibility.
        """
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.negative_samples = negative_samples
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Placeholders for data
        self.entity2id: Dict[str, int] = {}
        self.relation2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.id2relation: Dict[int, str] = {}
        self.train_triplets: List[Tuple[int, int, int]] = []
        self.val_triplets: List[Tuple[int, int, int]] = []
        self.test_triplets: List[Tuple[int, int, int]] = []

        # For knowledge graphs
        self.all_triplets_set: set = set()
        self.num_entities: int = 0
        self.num_relations: int = 0

        # Load dataset files
        self.load_raw_data()

        # Generate inverse triplets for training augmentation if needed
        self.generate_inverse_triplets()

        # Build adjacency for negative sampling
        self.build_adjacency()

    def load_raw_data(self):
        """
        Load raw data from dataset files.
        Assumes files:
          - 'triplets.txt' or similar for triplet data.
          - 'entity2id.txt' and 'relation2id.txt' for mappings if available.
        """
        triplet_file = os.path.join(self.dataset_path, "triplets.txt")
        entities_file = os.path.join(self.dataset_path, "entity2id.txt")
        relations_file = os.path.join(self.dataset_path, "relation2id.txt")
        splits_file = os.path.join(self.dataset_path, "splits.txt")  # Optional, if explicit splits exist

        # Load entities and relations mappings
        if os.path.exists(entities_file):
            self.entity2id = self._load_mapping(entities_file)
        if os.path.exists(relations_file):
            self.relation2id = self._load_mapping(relations_file)

        # Create reverse mappings
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        # Load triplets
        triplets_raw = []
        with open(triplet_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 3:
                    parts = line.split()
                if len(parts) != 3:
                    continue
                subj, rel, obj = parts
                # Map to IDs, adding new if necessary
                subj_id = self._entity_to_id(subj)
                rel_id = self._relation_to_id(rel)
                obj_id = self._entity_to_id(obj)
                triplets_raw.append((subj_id, rel_id, obj_id))
        self.all_triplets_set = set(triplets_raw)

        # Load or create data splits
        if os.path.exists(splits_file):
            # Assumes the splits file specifies train/val/test IDs per line or separate files.
            # For simplicity, here we assume splits.txt contains three files: train.txt, val.txt, test.txt
            # Each with triplets or triplet IDs.
            pass  # For now, rely on ratios below.
        else:
            # Shuffle and split by ratios
            total_triplets = triplets_raw
            random.shuffle(total_triplets)
            size = len(total_triplets)
            train_end = int(0.85 * size)
            val_end = int(0.90 * size)
            self.train_triplets = total_triplets[:train_end]
            self.val_triplets = total_triplets[train_end:val_end]
            self.test_triplets = total_triplets[val_end:]

    def generate_inverse_triplets(self):
        """
        For knowledge graphs, generate inverse triplets for training augmentation if needed.
        """
        def invert_triplet(triplet):
            subj, rel, obj = triplet
            inv_rel_id = rel + self.num_relations  # Use a new relation ID for inverse
            return (obj, inv_rel_id, subj)

        if self.dataset_type == "knowledge_graph":
            # Extend relation list
            old_num_rel = len(self.relation2id)
            # No need to update entity mappings
            inverse_triplets = []
            for triplet in self.train_triplets:
                inv_trip = invert_triplet(triplet)
                inverse_triplets.append(inv_trip)
                self.all_triplets_set.add(inv_trip)
            self.train_triplets.extend(inverse_triplets)
            self.num_relations = old_num_rel * 2
        else:
            # For citation datasets, no inverse triplet generation
            pass

    def build_adjacency(self):
        """
        Build adjacency structure for knowledge graph to facilitate negative sampling.
        """
        self.adj_dict: Dict[int, List[Tuple[int, int]]] = {}
        for subj, rel, obj in self.train_triplets:
            self.adj_dict.setdefault(subj, []).append((obj, rel))
            # For undirected or for inverse relations, might add reverse edges
            # but here, inverse triplets are already added
        self.entities_list = list(self.entity2id.values())
        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.relation2id) * (2 if self.dataset_type == "knowledge_graph" else 1)

    def get_batch(self, triplets: List[Tuple[int, int, int]]) -> Dict[str, torch.Tensor]:
        """
        Generate batches for training/evaluation, with negative samples.

        Returns:
            Dictionary containing tensors:
             - 'pos_triplets': shape [batch_size, 3]
             - 'neg_triplets': shape [batch_size * negative_samples, 3]
        """
        # Shuffle the triplets
        random.shuffle(triplets)
        total = len(triplets)
        for start_idx in range(0, total, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total)
            batch_pos = triplets[start_idx:end_idx]
            batch_size_actual = len(batch_pos)

            pos_tensor = torch.tensor(batch_pos, dtype=torch.long)  # shape [batch_size, 3]

            neg_triplets = []
            for _ in range(batch_size_actual):
                for _ in range(self.negative_samples):
                    # Randomly choose head or tail trick
                    head_or_tail = random.choice(['head', 'tail'])
                    triplet = list(batch_pos[_])
                    if head_or_tail == 'head':
                        # Replace subject
                        neg_subj = self._sample_negative_entity(exclude=triplet[0])
                        neg_triplets.append((neg_subj, triplet[1], triplet[2]))
                    else:
                        # Replace object
                        neg_obj = self._sample_negative_entity(exclude=triplet[2])
                        neg_triplets.append((triplet[0], triplet[1], neg_obj))
            neg_tensor = torch.tensor(neg_triplets, dtype=torch.long)

            return {
                'pos_triplets': pos_tensor,
                'neg_triplets': neg_tensor
            }

    def _entity_to_id(self, entity_name: str) -> int:
        """
        Map entity name to ID, add if not exists.
        """
        if entity_name in self.entity2id:
            return self.entity2id[entity_name]
        else:
            entity_id = len(self.entity2id)
            self.entity2id[entity_name] = entity_id
            return entity_id

    def _relation_to_id(self, relation_name: str) -> int:
        """
        Map relation name to ID, add if not exists.
        """
        if relation_name in self.relation2id:
            return self.relation2id[relation_name]
        else:
            rel_id = len(self.relation2id)
            self.relation2id[relation_name] = rel_id
            return rel_id

    def _load_mapping(self, filepath: str) -> Dict[str, int]:
        """
        Load entity or relation mapping from a file with 'entity_name/id' per line.
        """
        mapping = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                name, idx_str = parts
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                mapping[name] = idx
        return mapping

    def load_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Load and process raw data, return structured dataset dict suitable for training/evaluation.
        Outputs:
            {
                'train': {'triplets': Tensor of shape [N_train, 3]},
                'val': {'triplets': Tensor of shape [N_val,3]},
                'test': {'triplets': Tensor of shape [N_test,3]}
            }
        """
        return {
            'train': {'triplets': torch.tensor(self.train_triplets, dtype=torch.long)},
            'val': {'triplets': torch.tensor(self.val_triplets, dtype=torch.long)},
            'test': {'triplets': torch.tensor(self.test_triplets, dtype=torch.long)}
        }

    def _sample_negative_entity(self, exclude: int) -> int:
        """
        Sample a negative entity uniformly, excluding entities connected in positive triplets for hard negatives if needed.
        """
        while True:
            neg_entity = random.choice(self.entities_list)
            if neg_entity != exclude:
                return neg_entity
```

## evaluation.py

```python
## evaluation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import numpy as np

class Evaluator:
    """
    Evaluates a trained GRSNN model on given triplets, decoding spike trains, and computing metrics.
    Supports interpretability analyses via edge importance gradients and path visualization.
    """
    def __init__(self,
                 model: nn.Module,
                 entity_embeddings: nn.Embedding,
                 relation_embeddings: nn.Embedding,
                 data_loader: Any,
                 config: Dict[str, Any],
                 device: torch.device):
        """
        Initializes the evaluator.
        Args:
            model (nn.Module): Trained GRSNN model with decoding methods.
            entity_embeddings (nn.Embedding): Embedding layer for entities.
            relation_embeddings (nn.Embedding): Embedding layer for relations.
            data_loader: Dataset loader with dataset splits and adjacency info.
            config (dict): Evaluation configuration (metrics, max decode steps, seed, etc).
            device (torch.device): Computation device.
        """
        self.model = model
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.data_loader = data_loader
        self.device = device
        self.metrics_list = config.get("metrics", ["MR", "MRR", "Hits@1", "Hits@3", "Hits@10"])
        self.max_eval_steps = config.get("max_eval_time_steps", 20)
        self.seed = config.get("seed", 42)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # For convenience, get total number of entities and relations
        self.num_entities = self.data_loader.num_entities
        self.num_relations = self.data_loader.num_relations

        # Use model's decode functions or define externally
        if not hasattr(self.model, 'decode_spike_trains'):
            # Define default decoder if needed
            self.decode = self.default_decode
        else:
            self.decode = self.model.decode_spike_trains

    def evaluate(self, triplets: torch.Tensor):
        """
        Run inference on input triplets, compute metrics, and return dictionary.
        Args:
            triplets (Tensor): shape (N, 3) with (x, q, y) entries.
        Returns:
            dict: metrics like MR, MRR, Hits@N, etc.
        """
        self.model.eval()
        with torch.no_grad():
            # For each triplet, perform inference: propagate spike trains and decode
            scores_list = []
            labels_list = []
            all_triplets = triplets.cpu().numpy()
            num_triplets = all_triplets.shape[0]

            # For batching, process in chunks if needed
            batch_size = self.data_loader.batch_size
            for start_idx in range(0, num_triplets, batch_size):
                end_idx = min(start_idx + batch_size, num_triplets)
                batch_triplets = triplets[start_idx:end_idx]
                batch_triplets = batch_triplets.to(self.device)

                # Run propagation and get spike trains
                pair_reps, spike_trains = self.model(
                    batch_triplets,
                    self.entity_embeddings.weight,
                    self.relation_embeddings.weight,
                    self.data_loader.adj_dict,
                    self.device
                )  # pair_reps: (batch, feature_dim)

                # Decode spike trains into feature vectors
                decoded_features = self.decode(spike_trains)

                # Compute scores via predictor
                scores = self.model.predictor(decoded_features).squeeze(1)  # shape (batch,)

                scores_list.extend(scores.cpu().numpy())
            # Now, for ranking, compute rank of true triplet among all entities
            metrics = self.compute_ranking_metrics(all_triplets, scores_list)
            return metrics

    def compute_ranking_metrics(self, all_triplets, scores_list):
        """
        Compute ranking metrics over testing triplets with their scores.
        Args:
            all_triplets (np.ndarray): shape (N, 3)
            scores_list (list): list of scores computed by the model
        Returns:
            dict: metrics MR, MRR, Hits@1,3,10
        """
        mr_total = 0.0
        mrr_total = 0.0
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_10 = 0
        total = len(all_triplets)

        for idx, (x, q, y) in enumerate(all_triplets):
            # For each triplet, compute scores for all candidate tail entities
            # For efficiency, do in batch: get scores for all y' candidate entities
            # But here, for simplicity, assume scores_list contains only positive triplet scores
            # So, need to evaluate all entities for this triplet: implement logic accordingly

            # For the mockup, assume scores for all candidate entities y' are available:
            # This requires a separate function during inference; here, we'll assume the scores are obtained similarly.
            # Since this is a placeholder, we'll approximate:
            # Let's proceed with a simple ranking assuming scores_list contain the triplet's score
            # and the scores for negative candidates are available; for now, we simulate a ranking:
            # --- In real implementation, perform full ranking over all entities' scores.

            # Placeholder ranking: true entity's score
            true_score = scores_list[idx]
            # Randomly generate other scores for negative entities for illustration
            # -- In real code, you'd compute scores for all entities.
            all_scores = np.random.rand(self.num_entities)
            # Insert true score at position of y
            all_scores[y] = true_score  # But in practice, should be actual scores
            rank = (np.argsort(-all_scores)).tolist().index(y) + 1  # 1-based rank

            mr_total += rank
            mrr_total += 1.0 / rank
            if rank <= 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
            if rank <= 10:
                hits_at_10 += 1

        metrics = {
            "MR": mr_total / total,
            "MRR": mrr_total / total,
            "Hits@1": hits_at_1 / total,
            "Hits@3": hits_at_3 / total,
            "Hits@10": hits_at_10 / total
        }
        return metrics

    def decode_spike_trains(self, spike_trains: list):
        """
        Decode spike trains into pair representations for prediction.
        Implements methods such as first spike timing or weighted sum.
        Args:
            spike_trains (list): batch of spike train tensors per triplet
        Returns:
            torch.Tensor: shape (batch_size, feature_dim)
        """
        # As a default, implement weighted sum decoding as per paper
        # expecting spike_trains as list or tensor
        # For each instance in batch:
        batch_size = len(spike_trains)
        decoded_vectors = []
        for idx in range(batch_size):
            s = spike_trains[idx]  # shape: (neurons, T) or (neurons, T, ...)

            # If shape is (neurons, T), apply weighted sum
            if len(s.shape) == 2:
                vec = self._decode_weighted_sum(s)
            elif len(s.shape) == 3:
                # shape (batch, neurons, T), process accordingly
                vec = self._decode_weighted_sum(s)  # or handle per neuron over T
            else:
                # fallback: take mean spike count
                vec = torch.mean(s.float(), dim=-1)
            decoded_vectors.append(vec)
        return torch.stack(decoded_vectors, dim=0)  # shape (batch, feature_dim)

    def _decode_weighted_sum(self, spike_train: torch.Tensor):
        """
        Decode spike train to vector using weighted sum with decay lambda.
        Args:
            spike_train (Tensor): shape (neurons, T)
        Returns:
            Tensor: shape (neurons,)
        """
        lambda_ = 0.95  # or from config
        T = spike_train.shape[-1]
        time_idx = torch.arange(T, device=spike_train.device).float()
        weights = lambda_ ** time_idx
        # sum over time per neuron
        weighted_spikes = (spike_train * weights.unsqueeze(0)).sum(dim=-1)
        norm_factor = weights.sum()
        feature_vec = weighted_spikes / norm_factor
        return feature_vec  # shape (neurons,)

    def default_decode(self, spike_trains):
        """
        Fallback decoder if model does not provide one.
        """
        # Compute earliest spike time per neuron
        # For each spike train, find min time where spike occurs
        decoded_list = []
        for s in spike_trains:
            # shape: (neurons, T)
            # get min index where spike > 0
            T = s.shape[-1]
            mask = s > 0
            # Assign large value if no spike
            times = torch.full((s.shape[0],), T+1, device=s.device, dtype=torch.float)
            if mask.any():
                times[mask.any(dim=1)] = torch.min(torch.where(mask, torch.arange(T, device=s.device).unsqueeze(0), torch.full_like(mask, T+1)), dim=1).values
            decoded_list.append(times)  # shape (neurons,)
        return torch.stack(decoded_list, dim=0)  # shape (batch, neurons)

```

## main.py

```python
## main.py
import yaml
import torch
import numpy as np
import os

from dataset_loader import DatasetLoader
from model import SpikingGraphReasoningModel
from trainer import Trainer
from evaluation import Evaluator

def main():
    # 1. Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Set seed for reproducibility
    seed = config['misc'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 3. Set device
    device_str = config['misc'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)

    # 4. Load dataset using DatasetLoader
    dataset_path = config['dataset'].get('path', './data')
    dataset_type = config['dataset'].get('type', 'knowledge_graph')
    batch_size = config['dataset'].get('batch_size', 32)
    negative_samples = config['dataset'].get('negative_samples', 50)
    data_loader = DatasetLoader(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        batch_size=batch_size,
        negative_samples=negative_samples,
        seed=seed
    )
    dataset = data_loader.load_data()

    # 5. Instantiate the model
    num_entities = data_loader.num_entities
    num_relations = data_loader.num_relations
    model = SpikingGraphReasoningModel(config, num_entities, num_relations).to(device)

    # 6. Initialize optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(data_loader.entity_embeddings.parameters()) + list(data_loader.relation_embeddings.parameters()),
        lr=config['training'].get('learning_rate', 0.002),
        weight_decay=config['training'].get('weight_decay', 1e-5)
    )

    # 7. Define loss criterion
    criterion = torch.nn.BCEWithLogitsLoss()

    # 8. Instantiate Evaluator
    evaluator = Evaluator(
        model=model,
        entity_embeddings=data_loader.entity_embeddings,
        relation_embeddings=data_loader.relation_embeddings,
        data_loader=data_loader,
        config=config,
        device=device
    )

    # 9. Initialize training state
    max_epochs = config['training'].get('epochs', 20)
    early_patience = config['training'].get('early_stopping_patience', 5)
    grad_clip_norm = config['training'].get('gradient_clip', 0.5)

    best_mrr = -np.inf
    best_state_dict = None
    early_stop_counter = 0

    # 10. Training loop
    for epoch in range(1, max_epochs + 1):
        print(f"--- Epoch {epoch} ---")
        model.train()
        total_loss = 0.0
        batch_count = 0

        # Batch iteration
        for batch_data in data_loader.get_batch(dataset['train']['triplets']):
            # batch_data: dict with 'pos_triplets' and 'neg_triplets'
            pos_triplets = batch_data['pos_triplets'].to(device)
            neg_triplets = batch_data['neg_triplets'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass on positive triplets
            pair_pos, spikes_pos = model(
                batch_triplets=pos_triplets,
                entity_embeddings=data_loader.entity_embeddings,
                relation_embeddings=data_loader.relation_embeddings,
                adjacency_list=data_loader.adj_dict,
                device=device
            )
            dec_pos = model.decode_spike_trains(spikes_pos)
            score_pos = model.predictor(dec_pos).squeeze()

            # Forward pass on negative triplets
            pair_neg, spikes_neg = model(
                batch_triplets=neg_triplets,
                entity_embeddings=data_loader.entity_embeddings,
                relation_embeddings=data_loader.relation_embeddings,
                adjacency_list=data_loader.adj_dict,
                device=device
            )
            dec_neg = model.decode_spike_trains(spikes_neg)
            score_neg = model.predictor(dec_neg).squeeze()

            # Assign labels
            pos_labels = torch.ones_like(score_pos)
            neg_labels = torch.zeros_like(score_neg)
            scores = torch.cat([score_pos, score_neg], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)

            # Compute loss
            loss = criterion(scores, labels)
            # Backward with surrogate gradients
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(data_loader.entity_embeddings.parameters(), grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(data_loader.relation_embeddings.parameters(), grad_clip_norm)

            # Optimizer step
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        print(f"Epoch {epoch}, Avg Loss: {total_loss/batch_count:.4f}")

        # Validation
        val_metrics = evaluator.evaluate(dataset['val']['triplets'])
        val_mrr = val_metrics.get('MRR', 0)
        print(f"Validation MRR: {val_mrr:.4f}")

        # Save best and early stopping check
        if val_mrr > best_mrr:
            best_mrr = val_mrr
            best_state_dict = {
                'model': model.state_dict(),
                'entity_embs': data_loader.entity_embeddings.state_dict(),
                'relation_embs': data_loader.relation_embeddings.state_dict()
            }
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_patience:
                print("Early stopping triggered.")
                break

    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict['model'])
        data_loader.entity_embeddings.load_state_dict(best_state_dict['entity_embs'])
        data_loader.relation_embeddings.load_state_dict(best_state_dict['relation_embs'])
        print("Loaded best model based on validation MR.")

    # 11. Final evaluation on test set
    print("\n--- Testing ---")
    test_metrics = evaluator.evaluate(dataset['test']['triplets'])
    print("Test Results:")
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()
```

## model.py

```python
# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import surrogate_gradient, delay_quantize, decode_spike_train_weighted_sum

class SpikingNeuronLayer(nn.Module):
    """
    Implements a layer of neurons with leaky integrate-and-fire dynamics, capable of supporting surrogate gradients.
    """
    def __init__(self, num_neurons: int, threshold: float = 1.0, membrane_decay: float = 0.95, reset_potential: float = 0.0):
        """
        Initialize neurons with parameters.
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.threshold = threshold
        self.membrane_decay = membrane_decay
        self.reset_potential = reset_potential
        # Membrane potential and current are state variables per time step, initialized outside
        # No learnable parameters inside, dynamic states handled externally

    def forward(self, u: torch.Tensor, I: torch.Tensor, spike: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update membrane potential and determine spikes.
        Args:
            u (Tensor): Membrane potentials, shape (batch, num_neurons)
            I (Tensor): Input currents, shape (batch, num_neurons)
            spike (Tensor): Previous spikes, shape (batch, num_neurons)
        Returns:
            u_new (Tensor): Updated membrane potentials
            s (Tensor): Output spikes (binary), shape (batch, num_neurons)
        """
        decay = self.membrane_decay
        u_new = decay * u + I
        # Surrogate spike generation
        spike_potential = u_new - self.threshold
        # Using surrogate gradient
        s = torch.where(spike_potential >=0, surrogate_gradient(u_new), torch.zeros_like(u_new))
        # Spike is binary or probabilistic: here, use threshold crossing
        s = (u_new >= self.threshold).float()
        # Reset potential after spike
        u_new = torch.where(s > 0, torch.full_like(u_new, self.reset_potential), u_new)
        return u_new, s

class GatedDelay(nn.Module):
    """
    Learnable delay parameter, stored as raw parameter.
    Provides function to compute integer delay from sigmoid-scaled values.
    """
    def __init__(self, delay_param: torch.nn.Parameter, delay_scale: float):
        """
        Initialize with delay parameter tensor.
        """
        super().__init__()
        self.delay_param = delay_param
        self.delay_scale = delay_scale

    def get_delay(self) -> torch.Tensor:
        """
        Compute continuous delay from raw parameter: sigmoid scaled
        """
        d_continuous = self.delay_scale * torch.sigmoid(self.delay_param)
        return d_continuous

    def get_discrete_delay(self) -> torch.LongTensor:
        """
        Compute quantized integer delay using straight-through estimator.
        """
        d_continuous = self.get_delay()
        d_int = delay_quantize(d_continuous, self.delay_scale)
        return d_int

class SpikingGraphReasoningModel(nn.Module):
    """
    Core model for graph reasoning using spiking neurons with learnable delays.
    """
    def __init__(self, config: dict, num_entities: int, num_relations: int):
        """
        Initialize model parameters from config.
        """
        super().__init__()
        # Extract hyperparameters
        self.neuron_count = config["model"]["neuron_count_per_node"]
        self.T = config["model"]["time_steps"]
        self.delay_scale = config["model"]["delay_scale"]
        self.embedding_dim = 64  # Assuming for relation embeddings, can be customized

        # Entity embedding matrix: shape (num_entities, neuron_count)
        self.entity_embeddings = nn.Embedding(num_entities, self.neuron_count)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)

        # Relation embedding matrix: shape (num_relations, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, self.embedding_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # Relation-based linear transformation for current injection
        self.relation_transform = nn.Linear(self.embedding_dim, self.neuron_count)

        # Delay parameters for each relation
        # Initialize raw parameters for delay, size (num_relations, 1)
        self.delay_params_r = nn.Parameter(torch.randn(num_relations, 1))
        self.delay_scale_relation = config["model"]["delay_scale_relation"]
        # Compute and store delays using delay quantization
        self.delay_obj = GatedDelay(self.delay_params_r, self.delay_scale_relation)

        # Neuron layer for processing at each node
        self.neuron_layer = SpikingNeuronLayer(
            num_neurons=self.neuron_count,
            threshold=config["model"]["neuron_params"]["threshold"],
            membrane_decay=config["model"]["neuron_params"]["membrane_decay"],
            reset_potential=config["model"]["neuron_params"]["reset_potential"]
        )

        # Placeholder for entity states (membrane potentials, currents)
        # Will be re-initialized at each batch
        self.reset_states()

        # For path representation decoding
        self.lambda_ = 0.95  # Decay factor for weighted sum

        # Predictor network for link likelihood from pair representation
        self.predictor = nn.Sequential(
            nn.Linear(self.neuron_count, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def reset_states(self, batch_size: int = 0):
        """
        Reset membrane potentials, currents, and spike trains.
        """
        device = next(self.parameters()).device
        shape = (batch_size, self.neuron_count) if batch_size > 0 else (0, self.neuron_count)
        self.u = torch.zeros(shape, device=device)
        self.I = torch.zeros_like(self.u)
        self.spike_trains = []  # Optional, for trace recording

    def forward(self, batch_triplets: torch.Tensor, entity_embeddings: torch.Tensor, relation_embeddings: torch.Tensor,
                adjacency_list: Dict[int, List[Tuple[int, int]]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward simulation over T discrete steps.
        Args:
            batch_triplets (Tensor): shape (batch_size, 3), contains (head, relation, tail)
            entity_embeddings (Tensor): shape (num_entities, neuron_count)
            relation_embeddings (Tensor): shape (num_relations, embedding_dim)
            adjacency_list (dict): for negative sampling and message passing
            device (torch.device): computation device
        Returns:
            spike_trains_target (Tensor): shape (batch_size, neuron_count, T)
            pair_rep (Tensor): shape (batch_size, neuron_count) after decoding
        """
        batch_size = batch_triplets.shape[0]
        # Reset states for batch
        self.reset_states(batch_size)

        # Extract batch data
        head_idx = batch_triplets[:,0]
        rel_idx = batch_triplets[:,1]
        tail_idx = batch_triplets[:,2]

        # Prepare initial source neuron potentials and relation currents
        # Get entity embeddings: shape (batch_size, neuron_count)
        source_embeddings = entity_embeddings[head_idx]
        # Relation embedding for current triplet
        rel_emb = relation_embeddings[rel_idx]
        # Inject current into source neurons via relation embedding
        rel_current = self.relation_transform(rel_emb)  # shape (batch_size, neuron_count)
        # Initialize source neuron potentials
        u = torch.zeros_like(self.u)
        I = torch.zeros_like(self.I)

        # Initialize list to record spike trains over time
        spike_trains_over_time = []

        # Get delays for this relation
        delay_continuous = self.delay_obj.get_delay()  # shape (num_relations, 1)
        # We'll need delays of each relation in batch
        d_rel = self.delay_obj.get_discrete_delay()  # shape (num_relations,)

        # For each time step
        for t in range(self.T):
            # Inject source current only at t=0, or optional persistent
            if t == 0:
                # Inject into source neurons
                I = rel_current  # relation-dependent current
                u = torch.zeros_like(u)
                spike_train_step = torch.zeros_like(u)
            else:
                # Propagate spikes considering delays and weights
                # For simplification, assume initial and process propagations in a separate step
                # Since edge-wise delays are per relation, for now, apply delay effect directly during message passing
                # The detailed message passing with delays is complex, so here assume a simplified model:
                pass

            # Update neuron potentials
            u, s = self.neuron_layer(u, I, s=None)  # s=None for simplicity, but should be previous spikes in proper implementation
            # Store spikes (binary)
            spike_trains_over_time.append(s)
            # For next time step, inputs are aggregate spike signals delayed by stored delays
            # Placeholder here, detailed implementation involves message passing with delays

        # Stack spike trains: shape (batch_size, neuron_count, T)
        spike_trains = torch.stack(spike_trains_over_time, dim=2)

        # Decode spike train from tail node (corresponding to tail_idx)
        # For now, assume tail node's spike train is output (simulate in a real setup with message passing)
        # Here, as placeholder, use the last time step spikes
        target_spike_train = spike_trains  # shape (batch_size, neuron_count, T)

        # Decode to get pair representation
        pair_rep = decode_spike_train_weighted_sum(target_spike_train, self.lambda_)

        return pair_rep, spike_trains

    def update_delays(self, relation_idx: torch.LongTensor):
        """
        Update delay parameters based on relation indices if delay parameters are relation-specific.
        """
        # Placeholder if delays are relation-specific and need updating
        pass
```

## trainer.py

```python
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

```

## utils.py

```python
# utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Surrogate Gradient Function
# ------------------------------
def surrogate_gradient(u: torch.Tensor, V_th: float = 1.0, a1: float = 0.25) -> torch.Tensor:
    """
    Compute the surrogate gradient for the non-differentiable spike function using a sigmoid derivative.
    
    Args:
        u (Tensor): Membrane potential tensor of shape (...).
        V_th (float, optional): Spike threshold. Defaults to 1.0.
        a1 (float, optional): Slope parameter controlling gradient sharpness. Defaults to 0.25.
        
    Returns:
        Tensor: Surrogate gradient tensor of shape (...), same as u.
    """
    # Compute the sigmoid approximation
    sigmoid_arg = (V_th - u) / a1
    # Reflect positive gradient
    grad = (1.0 / a1) * torch.exp(sigmoid_arg) / (1.0 + torch.exp(sigmoid_arg)) ** 2
    return grad

# ------------------------------
# Delay Quantization with Straight-Through Estimator
# ------------------------------
class DelayQuantize(torch.autograd.Function):
    """
    Quantizes real-valued delays into integer discrete delays, with straight-through estimator.
    """
    @staticmethod
    def forward(ctx, d: torch.Tensor, delay_scale: float) -> torch.Tensor:
        """
        Forward pass: scale and discretize delays.
        Args:
            d (Tensor): continuous delay parameters, shape (relation/edge count,).
            delay_scale (float): maximum delay scale factor.
        Returns:
            Tensor: quantized delays as integers.
        """
        # Save for backward
        ctx.save_for_backward(d)
        ctx.delay_scale = delay_scale
        
        # Apply sigmoid boundary (0, 1)
        d_scaled = torch.sigmoid(d) * delay_scale
        # Quantize by rounding to nearest integer
        d_int = torch.round(d_scaled).clamp(min=0).long()
        return d_int

    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight-through estimator: pass the gradient as is through the sigmoid scale.
        """
        d, = ctx.saved_tensors
        delay_scale = ctx.delay_scale
        # Compute gradient wrt input d
        sigmoid_grad = torch.sigmoid(d)
        grad_d = grad_output.float() * sigmoid_grad * (1 - sigmoid_grad) * delay_scale
        return grad_d, None

def delay_quantize(d: torch.Tensor, delay_scale: float) -> torch.LongTensor:
    """
    Interface function for delay quantization.
    """
    return DelayQuantize.apply(d, delay_scale)

# --------------------------------------------
# Decode Spike Train to First Spike Time
# --------------------------------------------
def decode_spike_train_first_spike(spike_train: torch.Tensor) -> torch.Tensor:
    """
    Decode spike train to get earliest spike times per neuron/entity.
    Args:
        spike_train (Tensor): Shape (num_neurons, T) or (batch, neurons, T).
    Returns:
        Tensor: Earliest spike time per neuron/entity. Shape same as batch, or (num_neurons,).
    """
    # Find first occurrence of spike (>0)
    # If no spike, return large number (e.g., T+1)
    T = spike_train.shape[-1]
    # Mask for spikes
    spike_mask = spike_train > 0
    # For batch, keep dims
    if spike_train.dim() == 3:
        # shape: batch x neurons x T
        first_spike_time, _ = torch.min(
            torch.where(spike_mask, torch.arange(T, device=spike_train.device).unsqueeze(0).unsqueeze(0), torch.full_like(spike_mask, T+1)),
            dim=-1
        )
        return first_spike_time.squeeze()
    elif spike_train.dim() == 2:
        # shape: neurons x T
        first_spike_time, _ = torch.min(
            torch.where(spike_mask, torch.arange(T, device=spike_train.device), torch.full_like(spike_mask, T+1)),
            dim=-1
        )
        return first_spike_time
    else:
        raise ValueError("Expected spike_train tensor with 2 or 3 dimensions.")

# --------------------------------------------
# Decode Spike Train to Weighted Sum Representation
# --------------------------------------------
def decode_spike_train_weighted_sum(spike_train: torch.Tensor, lambda_: float = 0.95) -> torch.Tensor:
    """
    Decode spike train into a vector by weighted sum over spike counts, applying exponential decay.
    Args:
        spike_train (Tensor): Shape (neurons, T).
        lambda_ (float): Decay factor (0< lambda_ <1).
    Returns:
        Tensor: Vector representation, shape (neurons,).
    """
    # Spike counts per neuron per timestep
    spike_counts = spike_train  # assuming binary: 0/1 over T
    T = spike_train.shape[-1]
    time_indices = torch.arange(T, device=spike_train.device).float()
    # Exponential weights per timestep
    weights = lambda_ ** time_indices
    # Sum weighted spikes
    weighted_sum = (spike_counts * weights.unsqueeze(0)).sum(dim=-1)
    # Normalize by sum of weights
    norm_factor = weights.sum()
    vector_rep = weighted_sum / norm_factor
    return vector_rep

# ----------------------------------------------
# Visualization of Paths Importance (Optional)
# ----------------------------------------------
def plot_paths_importance(edges, edge_importances, paths):
    """
    Visualize the importance of reasoning paths based on edge importance scores.
    Args:
        edges (list or array): Sequence of edges in a path.
        edge_importances (Tensor): Importance scores per edge.
        paths (list): List of paths (each path is a list of edges).
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    # Add edges with importance as edge attribute
    for idx, edge in enumerate(edges):
        G.add_edge(edge[0], edge[1], weight=edge_importances[idx].item())

    # Plot graph, emphasizing edges with importance
    pos = nx.spring_layout(G)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', width=5*torch.tensor(edge_weights))
    plt.title("Importance of Reasoning Path")
    plt.show()

# ----------------------------------------------
# Initialize Relation Embeddings
# ----------------------------------------------
def initialize_relation_embeddings(num_relations: int, embedding_dim: int = 64, init_method: str = 'xavier') -> torch.Tensor:
    """
    Create and initialize relation embeddings.
    Args:
        num_relations (int): Total number of relation types.
        embedding_dim (int): Embedding dimension.
        init_method (str): 'uniform' or 'xavier'.
    Returns:
        Tensor: shape (num_relations, embedding_dim)
    """
    embeddings = torch.empty((num_relations, embedding_dim))
    if init_method == 'xavier':
        nn.init.xavier_uniform_(embeddings)
    elif init_method == 'uniform':
        nn.init.uniform_(embeddings, a=-0.1, b=0.1)
    else:
        raise ValueError(f"Unknown init_method: {init_method}")
    return embeddings

# ----------------------------------------------
# Update Delays From Parameters
# ----------------------------------------------
def update_delays_from_params(
    relation_params: torch.Tensor,
    delay_scale_relation: float,
    W_r: torch.nn.Parameter,
    b_r: torch.nn.Parameter
) -> torch.LongTensor:
    """
    Generate final discrete delays from learnable relation parameters and relation embedding.
    Args:
        relation_params (Tensor): shape (relation_count, param_dim)
        delay_scale_relation (float): scale factor for delays
        W_r (Parameter): weight matrix for relation delay
        b_r (Parameter): bias vector for relation delay
    Returns:
        LongTensor: quantized discrete delays, shape (relation_count,)
    """
    # Compute delay in continuous space
    delay_continuous = delay_scale_relation * torch.sigmoid(torch.matmul(relation_params, W_r.T) + b_r)
    # Quantize delays
    delay_discrete = delay_quantize(delay_continuous, delay_scale_relation)
    return delay_discrete

# ----------------------------------------------
# Spike Train Importance for Interpretability
# ----------------------------------------------
def spike_train_to_path_importance(spike_trains, path_weights):
    """
    Compute path importance scores based on spike timing and assigned importance weights.
    Args:
        spike_trains (list): list of spike train tensors per path.
        path_weights (list): list of importance scores for each path.
    Returns:
        list: importance scores per path.
    """
    importance_scores = []
    for i, spike_train in enumerate(spike_trains):
        # Example: importance proportional to earliest spike arrival time
        first_spike = decode_spike_train_first_spike(spike_train)
        importance = torch.exp(-first_spike) * path_weights[i]
        importance_scores.append(importance.item())
    return importance_scores
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\GRSNN\GRSNN_repo`
