## utils.py
import numpy as np
import torch
import yaml
import logging
import sys
import matplotlib.pyplot as plt
import networkx as nx

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cosine_similarity(vec1, vec2, epsilon: float = 1e-6):
    """
    Compute cosine similarity between two vectors or batches of vectors.
    Args:
        vec1 (np.ndarray or torch.Tensor): First vector or batch of vectors.
        vec2 (np.ndarray or torch.Tensor): Second vector or batch of vectors.
        epsilon (float): Small value to prevent division by zero.
    Returns:
        float or np.ndarray: Cosine similarity score(s).
    """
    # Convert inputs to numpy arrays if they are torch tensors
    if isinstance(vec1, torch.Tensor):
        vec1 = vec1.detach().cpu().numpy()
    if isinstance(vec2, torch.Tensor):
        vec2 = vec2.detach().cpu().numpy()
    # Normalize vectors
    vec1_norm = np.linalg.norm(vec1, axis=-1, keepdims=True) + epsilon
    vec2_norm = np.linalg.norm(vec2, axis=-1, keepdims=True) + epsilon
    vec1_normalized = vec1 / vec1_norm
    vec2_normalized = vec2 / vec2_norm
    # Compute cosine similarity
    similarity = np.sum(vec1_normalized * vec2_normalized, axis=-1)
    # Clip to [-1, 1]
    similarity = np.clip(similarity, -1.0, 1.0)
    if similarity.shape == ():  # single pair
        return float(similarity)
    return similarity

def normalize(vectors: np.ndarray, axis: int = 1, epsilon: float = 1e-6) -> np.ndarray:
    """
    Normalize vectors along a specified axis to unit length.
    Args:
        vectors (np.ndarray): Input array of vectors.
        axis (int): Axis along which to normalize.
        epsilon (float): Small value to prevent division by zero.
    Returns:
        np.ndarray: Normalized vectors.
    """
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True) + epsilon
    return vectors / norms

def plot_graph(graph: nx.Graph, highlighted_nodes=None, title: str = ""):
    """
    Plot a NetworkX graph with optional highlighted nodes.
    Args:
        graph (nx.Graph): The graph to visualize.
        highlighted_nodes (list): Nodes to highlight in color.
        title (str): Plot window title.
    """
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)

    # Default node color
    node_colors = ['lightblue' if node not in highlighted_nodes else 'orange' for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=300, alpha=0.8)
    nx.draw_networkx_edges(graph, pos, arrows=True, alpha=0.5)
    labels = {n: str(n) for n in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels, font_size=9)

    if highlighted_nodes:
        nx.draw_networkx_nodes(graph, pos, nodelist=highlighted_nodes, node_color='red', node_size=400)

    plt.title(title)
    plt.axis('off')
    plt.show()

def load_config(filepath: str = 'config.yaml') -> dict:
    """
    Load and parse the YAML configuration file.
    Args:
        filepath (str): Path to the configuration YAML.
    Returns:
        dict: Parsed configuration dictionary.
    """
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict), "Configuration file must contain a dictionary at top level."
        return config
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        sys.exit(1)

def pretty_print_dict(d: dict, indent: int = 2):
    """
    Utility to pretty-print dictionaries for debugging.
    Args:
        d (dict): Dictionary to print.
        indent (int): Indentation level.
    """
    import json
    print(json.dumps(d, indent=indent))

def save_figure(fig: plt.Figure, path: str):
    """
    Save a matplotlib figure to the specified path.
    Args:
        fig (plt.Figure): The figure object.
        path (str): File path to save.
    """
    try:
        fig.savefig(path, bbox_inches='tight')
        logger.info(f"Figure saved to {path}")
    except Exception as e:
        logger.warning(f"Failed to save figure to {path}: {e}")

# Optional: Additional utility functions for text processing or debugging can be added here
# e.g., text cleaning, token count estimation, or answer text normalization, as needed.
