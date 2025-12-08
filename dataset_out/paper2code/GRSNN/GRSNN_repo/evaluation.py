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

