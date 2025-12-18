## visualization.py
"""
This module provides functions to visualize ontologies (graphs) with edges colored or styled
based on confidence or similarity scores, aiding qualitative analysis.

Dependencies:
- networkx
- matplotlib.pyplot
- numpy
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# If using color maps
from matplotlib import cm

# To ensure compatibility with Graph class or networkx graphs
from typing import Optional, Dict, List, Union

def visualize_graph(
    graph: "Graph",
    output_path: str,
    title: str = "Ontology Graph Visualization",
    node_labels: Optional[Dict[str, str]] = None,
    edge_labels: Optional[Dict[Tuple[str, str], float]] = None,
    edge_scores: Optional[Dict[Tuple[str, str], float]] = None,
    score_normalization: str = "auto",
    colormap_name: str = "viridis",
    node_size: int = 300,
    font_size: int = 10,
    dpi: int = 300,
    with_arrow: bool = True,
    show_plot: bool = False
) -> None:
    """
    Visualize a directed graph with edges colored according to their scores (confidence or similarity).

    Args:
        graph: The Graph object to visualize. Assumed to have 'nodes' attribute and edges with 'weight' or 'score'.
        output_path: Path to save the figure (e.g., PNG, PDF).
        title: Plot title.
        node_labels: Optional dict mapping node to label text. Defaults to node string if None.
        edge_labels: Optional dict mapping edge to label (e.g., score). Used for annotation.
        edge_scores: Optional dict mapping edge tuple to score for coloring; if None, use 'weight' attribute.
        score_normalization: 'auto', 'minmax', or 'percentile'; determines how to scale scores for color mapping.
        colormap_name: Name of matplotlib colormap to use.
        node_size: Size of nodes.
        font_size: Font size for labels.
        dpi: Resolution of saved figure.
        with_arrow: Whether to draw arrows (directed edges).
        show_plot: Whether to display the plot interactively (default False, just saves figure).

    Outputs:
        None, but saves figure at output_path.
    """
    # Convert custom Graph to networkx DiGraph if needed
    # Assuming graph is either of class 'Graph' or nx.DiGraph
    if hasattr(graph, 'nodes'):
        G = graph
        if hasattr(G, 'edges'):
            nx_graph = nx.DiGraph()
            for u in G.nodes:
                nx_graph.add_node(u)
            for u, v, d in getattr(G, 'edges', G).edges(data=True):
                weight = d.get('weight', 1.0)
                nx_graph.add_edge(u, v, weight=weight)
        else:
            # Possibly already networkx graph
            nx_graph = G
    else:
        # Assume it's already a networkx DiGraph
        nx_graph = graph

    # Determine scores for coloring edges
    if edge_scores is not None:
        scores = list(edge_scores.values())
        # Map scores to [0,1] for colormap
        min_score = min(scores)
        max_score = max(scores)
        score_array = np.array(scores)
    else:
        # Use 'weight' attribute for edges
        scores = [d.get('weight', 1.0) for _, _, d in nx_graph.edges(data=True)]
        min_score = min(scores)
        max_score = max(scores)
        score_array = np.array(scores)

    # Normalize scores based on selected method
    if score_normalization == "auto":
        # Use min and max of scores
        norm_scores = (score_array - min_score) / (max_score - min_score + 1e-8)
    elif score_normalization == "minmax":
        norm_scores = (score_array - np.min(score_array)) / (np.max(score_array) - np.min(score_array) + 1e-8)
    elif score_normalization == "percentile":
        percentile_90 = np.percentile(score_array, 90)
        norm_scores = np.clip((score_array - np.min(score_array)) / (percentile_90 - np.min(score_array) + 1e-8), 0, 1)
    else:
        # Default: no normalization
        norm_scores = score_array

    # Map normalized scores to colormap
    cmap = cm.get_cmap(colormap_name)
    edge_colors = [cmap(score) for score in norm_scores]

    # Prepare layout
    # For small to medium graphs, spectral_layout or spring_layout is good
    pos = nx.spring_layout(nx_graph, seed=42, k=0.5)

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(nx_graph, pos, node_size=node_size, node_color='lightblue')
    if with_arrow:
        nx.draw_networkx_edges(nx_graph, pos, arrowstyle='-|>', arrowsize=15, edge_color=edge_colors, width=2)
    else:
        nx.draw_networkx_edges(nx_graph, pos, edge_color=edge_colors, width=2)

    # Draw labels
    labels = node_labels if node_labels else {n: n for n in nx_graph.nodes}
    nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_size=font_size)

    # Add edge labels if provided
    if edge_labels:
        # Create dict for edge labels
        nx.draw_networkx_edge_labels(
            nx_graph,
            pos,
            edge_labels=edge_labels,
            font_size=font_size * 0.8,
            label_pos=0.5,
            font_color='red'
        )

    # Add colorbar for edge scoring
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_score, vmax=max_score))
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.set_label('Edge Score', fontsize=12)

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


def plot_subgraph(
    graph: "Graph",
    center_nodes: List[str],
    radius: int,
    output_path: str,
    title: str = "Subgraph Visualization",
    node_labels: Optional[Dict[str, str]] = None,
    edge_scores: Optional[Dict[Tuple[str, str], float]] = None,
    node_color_map: Optional[Dict[str, str]] = None,
    node_highlight_color: str = 'orange',
    node_normal_color: str = 'lightgrey',
    edge_color: str = 'black',
    node_size: int = 300,
    font_size: int = 10,
    dpi: int = 300,
    show_plot: bool = False
) -> None:
    """
    Visualize a subgraph within specified hop distance from center nodes, highlighting centers.
    """
    # Convert to networkx DiGraph if needed
    if hasattr(graph, 'nodes'):
        G = nx.DiGraph()
        for u in graph.nodes:
            G.add_node(u)
        for u, v, d in getattr(graph, 'edges', graph).edges(data=True):
            weight = d.get('weight', 1.0)
            G.add_edge(u, v, weight=weight)
    else:
        G = graph

    # Extract subgraph within radius
    nodes_in_subgraph = set()
    for center in center_nodes:
        if center not in G:
            continue
        nodes_in_subgraph.update(nx.single_source_shortest_path_length(G, center, cutoff=radius).keys())
    subG = G.subgraph(nodes_in_subgraph)

    # Position
    pos = nx.spring_layout(subG, seed=42)

    # Node colors: highlight centers
    node_colors = []
    for n in subG.nodes:
        if n in center_nodes:
            node_colors.append(node_highlight_color)
        else:
            node_colors.append(node_normal_color)

    # Edge colors
    if edge_scores:
        # Map scores to colors
        scores = [edge_scores.get((u, v), 0.0) for u, v in subG.edges]
        min_score = min(scores)
        max_score = max(scores)
        cmap = cm.get_cmap('coolwarm')
        edge_colors = [cmap((sc - min_score) / (max_score - min_score + 1e-8)) for sc in scores]
    else:
        edge_colors = edge_color

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_size)
    nx.draw_networkx_edges(subG, pos, edge_color=edge_colors, width=2, arrows=True)
    labels = {n: n for n in subG.nodes}
    nx.draw_networkx_labels(subG, pos, labels=labels, font_size=font_size)

    # Optional: add edge labels for scores
    if edge_scores:
        edge_labels = {}
        for u, v in subG.edges:
            score = edge_scores.get((u, v), None)
            if score is not None:
                edge_labels[(u, v)] = f"{score:.2f}"
        nx.draw_networkx_edge_labels(
            subG,
            pos,
            edge_labels=edge_labels,
            font_size=font_size * 0.8,
            label_pos=0.5,
            font_color='red'
        )

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
