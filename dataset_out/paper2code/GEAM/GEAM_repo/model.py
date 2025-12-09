## model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

class EdgeEncoder(nn.Module):
    """
    Encodes edge features into a dense vector space.
    """
    def __init__(self, edge_feat_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, edge_attr):
        return self.mlp(edge_attr)

class GCNConv(MessagePassing):
    """
    Custom GCN layer that incorporates edge features.
    """
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super().__init__(aggr='add')  # "Add" aggregation.
        self.linear_node = nn.Linear(in_channels, out_channels)
        self.linear_edge = nn.Linear(edge_dim, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_node.weight)
        nn.init.xavier_uniform_(self.linear_edge.weight)

    def forward(self, x, edge_index, edge_attr):
        # Add self loops to include node's own features
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: neighbor node features
        return self.linear_node(x_j) + self.linear_edge(edge_attr)

    def update(self, aggr_out):
        return F.relu(aggr_out)

class GNNEncoder(nn.Module):
    """
    GNN encoder with multiple message passing steps for molecular graphs.
    """
    def __init__(self,
                 input_dim: int = 16,
                 hidden_dim: int = 128,
                 num_passes: int = 3,
                 fc_layers: int = 2,
                 edge_feat_dim: int = 6):
        """
        Args:
            input_dim: Dimension of input node features.
            hidden_dim: Dimension of node embeddings.
            num_passes: Number of message passing iterations.
            fc_layers: Number of dense layers after message passing.
            edge_feat_dim: Dimension of edge feature vector.
        """
        super().__init__()
        self.num_passes = num_passes
        self.edge_encoder = EdgeEncoder(edge_feat_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GCNConv(input_dim, hidden_dim, edge_dim=hidden_dim)
            for _ in range(num_passes)
        ])
        # Final FC layers for processing node features if needed
        fc_modules = []
        in_dim = hidden_dim
        for _ in range(fc_layers):
            fc_modules.append(nn.Linear(in_dim, hidden_dim))
            fc_modules.append(nn.ReLU())
            in_dim = hidden_dim
        self.final_fc = nn.Sequential(*fc_modules)

    def forward(self, data: Data):
        """
        Args:
            data: torch_geometric.data.Data object with attributes:
                - x: node features, shape [num_nodes, input_dim]
                - edge_index: connectivity
                - edge_attr: edge features, shape [num_edges, edge_feat_dim]
        Returns:
            Node embeddings: tensor of shape [num_nodes, hidden_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Encode edge features
        edge_emb = self.edge_encoder(edge_attr)

        # Run message passing layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_emb)

        # Optional: refine with additional FC layers
        node_embeddings = self.final_fc(x)

        return node_embeddings

    def fragment_pooling(self, node_embeddings: torch.Tensor, fragment_node_mask: torch.Tensor):
        """
        Compute mean pooling over nodes belonging to each fragment.
        
        Args:
            node_embeddings: shape [num_nodes, hidden_dim]
            fragment_node_mask: bool tensor shape [num_nodes], True for nodes in fragment
        Returns:
            Fragment embedding: shape [hidden_dim]
        """
        selected_nodes = node_embeddings[fragment_node_mask]
        if selected_nodes.shape[0] == 0:
            # No nodes in the fragment; return zero vector
            return torch.zeros(node_embeddings.size(1), device=node_embeddings.device)
        else:
            return selected_nodes.mean(dim=0)
