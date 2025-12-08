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
