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
