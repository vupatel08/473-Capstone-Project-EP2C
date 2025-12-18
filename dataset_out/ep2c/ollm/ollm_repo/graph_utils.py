## graph_utils.py
import networkx as nx
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

class GraphUtils:
    @staticmethod
    def create_graph_from_relations(
        relations: List[Tuple[str, str, str, Optional[float]]],
        default_weight: float = 1.0
    ) -> nx.DiGraph:
        """
        Constructs a directed graph from a list of relations.
        Each relation is a tuple: (concept1, relation_type, concept2, weight?)
        If weight is not provided, defaults to 1.0.
        """
        G = nx.DiGraph()
        for rel in relations:
            u, rel_type, v = rel[:3]
            weight = rel[3] if len(rel) > 3 and rel[3] is not None else default_weight
            G.add_node(u)
            G.add_node(v)
            # Use relation type as edge attribute
            if G.has_edge(u, v):
                G[u][v]['weight'] += weight
                # Optionally, could store relation types as set or list
                if 'types' in G[u][v]:
                    G[u][v]['types'].add(rel_type)
                else:
                    G[u][v]['types'] = {rel_type}
            else:
                G.add_edge(u, v, weight=weight, types={rel_type})
        return G

    @staticmethod
    def merge_graphs(graph_list: List[nx.DiGraph]) -> nx.DiGraph:
        """
        Merges multiple graphs into one by summing edge weights for overlapping edges.
        """
        merged = nx.DiGraph()
        for g in graph_list:
            for u, v, data in g.edges(data=True):
                weight = data.get('weight',1.0)
                if merged.has_edge(u, v):
                    merged[u][v]['weight'] += weight
                else:
                    merged.add_node(u)
                    merged.add_node(v)
                    merged.add_edge(u, v, weight=weight, types=data.get('types',set()))
                # Merge relation types
                if 'types' in merged[u][v]:
                    merged[u][v]['types'].update(data.get('types',set()))
        return merged

    @staticmethod
    def prune_edges(
        graph: nx.DiGraph,
        alpha: float = 0.5,
        beta: float = 0.5
    ) -> nx.DiGraph:
        """
        Prunes edges based on global weight quantile threshold (alpha)
        and local relative importance threshold (beta).
        """
        # Make a copy to avoid modifying original
        G = graph.copy()

        # 1. Absolute thresholding based on alpha quantile
        weights = [d['weight'] for u, v, d in G.edges(data=True)]
        if not weights:
            return G
        threshold_alpha = np.quantile(weights, alpha)
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold_alpha]
        G.remove_edges_from(edges_to_remove)

        # 2. Relative thresholding per node based on beta
        for u in list(G.nodes):
            out_edges = list(G.out_edges(u, data=True))
            if not out_edges:
                continue
            total_weight = sum(d['weight'] for _, _, d in out_edges)
            # Sort by weight ascending
            out_edges_sorted = sorted(out_edges, key=lambda x: x[2]['weight'])
            cumulative = 0.0
            for u2, v2, d2 in out_edges_sorted:
                cumulative += d2['weight']
                if total_weight > 0:
                    cumsum_ratio = cumulative / total_weight
                else:
                    cumsum_ratio = 0
                if cumsum_ratio <= beta:
                    continue  # Keep this edge
                else:
                    # Remove edges exceeding beta
                    # Remove edges with cumulative ratio > beta
                    # Because sorted ascending, break when cumsum exceeds threshold
                    # But since cumulative is increasing, we can remove from here
                    for _, v_del, d_del in out_edges_sorted[out_edges_sorted.index((u2,v2,d2))+1:]:
                        G.remove_edge(u, v_del)
                    break
        # 3. Remove isolated or empty nodes
        isolated_nodes = [n for n in G.nodes if G.degree(n) == 0]
        G.remove_nodes_from(isolated_nodes)
        return G

    @staticmethod
    def remove_self_loops(graph: nx.DiGraph) -> nx.DiGraph:
        """
        Removes all self-loop edges in the graph.
        """
        G = graph.copy()
        self_loops = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loops)
        return G

    @staticmethod
    def remove_inverse_edges(graph: nx.DiGraph) -> nx.DiGraph:
        """
        For bidirectional edges between two nodes, keep only one arrow:
        remove the one with lower weight.
        """
        G = graph.copy()
        edges_checked = set()
        for u, v in G.edges():
            if (v, u) in G.edges() and (v, u) not in edges_checked:
                w_uv = G[u][v]['weight']
                w_vu = G[v][u]['weight']
                if w_vu > w_uv:
                    G.remove_edge(u, v)
                else:
                    G.remove_edge(v, u)
                edges_checked.add((u, v))
                edges_checked.add((v, u))
        return G

    @staticmethod
    def remove_cycles(graph: nx.DiGraph, strategy: str='greedy') -> nx.DiGraph:
        """
        Removes edges to make graph acyclic based on strategy.
        Default strategy 'greedy' removes edges involved in cycles, starting from smallest weight.
        """
        G = graph.copy()
        try:
            cycles = list(nx.simple_cycles(G))
        except nx.NetworkXNoCycle:
            cycles = []
        while cycles:
            for cycle in cycles:
                # Find the edge with the minimal weight in this cycle
                edges_in_cycle = [(cycle[i], cycle[(i+1)%len(cycle)]) for i in range(len(cycle))]
                edges_w = [G[u][v]['weight'] for u, v in edges_in_cycle if G.has_edge(u, v)]
                min_idx = np.argmin(edges_w)
                u_min, v_min = edges_in_cycle[min_idx]
                G.remove_edge(u_min, v_min)
                break
            try:
                cycles = list(nx.simple_cycles(G))
            except nx.NetworkXNoCycle:
                break
        return G

    @staticmethod
    def visualize_graph(
        graph: nx.DiGraph,
        output_path: str,
        options: Optional[Dict] = None
    ) -> None:
        """
        Visualizes the graph using Matplotlib.
        options can include node size, edge width, with labels, etc.
        """
        G = graph
        plt.figure(figsize=options.get('figsize', (12,8)) if options else (12,8))
        pos = nx.spring_layout(G, k=0.5)
        edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
        max_weight = max(edge_weights) if edge_weights else 1
        # Normalize edge width for visualization
        edges = G.edges()
        widths = [ (d['weight']/max_weight)*3 for u, v, d in G.edges(data=True)]
        nx.draw(G, pos,
                with_labels=options.get('with_labels', True) if options else True,
                node_size=options.get('node_size', 300) if options else 300,
                arrows=True,
                width=widths,
                edge_color=options.get('edge_color', 'gray') if options else 'gray'
        )
        plt.title(options.get('title', 'Graph Visualization') if options else 'Graph')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
