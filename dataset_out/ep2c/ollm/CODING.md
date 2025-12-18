# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py

import os
import json
import requests
import csv
from collections import deque, defaultdict
from typing import List, Dict, Tuple
import hashlib

# Import Dataset class as specified in the Data Structures and Interfaces
# For illustration, define a simple Dataset data class here
from dataclasses import dataclass

@dataclass
class Dataset:
    documents: List[str]
    concepts: List[str]
    relations: List[Tuple[str, str, str]]  # (concept1, relation_type, concept2)
    annotations: Dict[int, List[str]]     # document_id -> list of concepts


class DatasetLoader:
    def __init__(self, config: Dict):
        self.config = config
        # Directory to cache datasets
        self.cache_dir = "cached_datasets"
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_wikipedia(self) -> Dataset:
        """
        Load and process Wikipedia dataset:
        - Perform BFS from 'Main topic classifications' category up to depth 3.
        - Retrieve page titles and summaries for concepts.
        - Collect documents annotated with concepts.
        """
        cache_path = os.path.join(self.cache_dir, "wikipedia_dataset.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            documents = data['documents']
            concepts = data['concepts']
            relations = data['relations']
            annotations = {int(k): v for k, v in data['annotations'].items()}
            return Dataset(documents, concepts, relations, annotations)

        # Step 1: Retrieve categories starting from 'Main topic classifications'
        starting_category = "Main topic classifications"
        max_depth = 3
        category_graph, category_to_id, id_to_category = self._build_category_graph_bfs(starting_category, max_depth)

        # Step 2: Gather pages and summaries for each category
        concepts = list(category_to_id.keys())
        concept_id_map = {c: category_to_id[c] for c in concepts}

        # For each category, get page titles and summaries
        category_pages = self._get_category_pages(concept_to_id=category_to_id, max_pages=5000)

        # Create documents: concatenate title and summary
        documents = []
        annotations = defaultdict(list)  # document index -> list of concepts
        for cat_id, pages in category_pages.items():
            for idx, page in enumerate(pages):
                doc_text = self._combine_title_summary(page['title'], page['summary'])
                documents.append(doc_text)
                # Assign concepts based on category
                annotations[len(documents)-1].append(cat_id)  # Using category id as concept; can map back to name if needed

        # Build relations: parent-child among categories
        relations = []
        for parent_id, child_id in category_graph:
            relations.append((parent_id, "is-a", child_id))
        # Remove duplicates
        relations = list(set(relations))
        # Convert concept IDs back to names
        concepts = list(concept_to_id.keys())

        # Save cache
        data_to_cache = {
            'documents': documents,
            'concepts': concepts,
            'relations': relations,
            'annotations': dict(annotations)
        }
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_cache, f)

        return Dataset(documents, concepts, relations, dict(annotations))

    def load_arxiv(self) -> Dataset:
        """
        Load and process arXiv dataset:
        - Filter papers from 2020-2022 with ≥10 citations.
        - Text from title + abstract.
        - Concepts from arXiv taxonomy/keywords.
        - Map documents to concepts.
        """
        cache_path = os.path.join(self.cache_dir, "arxiv_dataset.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            documents = data['documents']
            concepts = data['concepts']
            relations = data['relations']
            annotations = {int(k): v for k, v in data['annotations'].items()}
            return Dataset(documents, concepts, relations, annotations)

        # Step 1: Load dataset from arXiv (assumed preprocessed locally)
        arxiv_metadata_path = self.config.get('arxiv_metadata_path', 'arxiv_metadata.csv')
        # The CSV should contain at least: paper_id, title, abstract, submission_date, citation_count, primary_categories
        documents = []
        concept_set = set()
        doc_annotations = defaultdict(list)
        with open(arxiv_metadata_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Filter criteria
                year = int(row['submission_date'][:4])
                citations = int(row.get('citation_count', 0))
                if 2020 <= year <= 2022 and citations >= 10:
                    text = f"{row['title']} {row['abstract']}"
                    documents.append(text)
                    # Generate concept(s) from primary categories or keywords
                    concepts_for_doc = self._extract_concepts_from_categories(row['primary_categories'])
                    for c in concepts_for_doc:
                        concept_set.add(c)
                    # Save annotations
                    doc_annotations[len(documents)-1] = concepts_for_doc

        concepts = list(concept_set)

        # Build relations: for illustration, assume hierarchical relation 'is-a' among concepts
        relations = []
        # This step can be extended based on actual arXiv taxonomy
        # For simplicity, treat primary categories as concepts linked to broader categories
        # Alternatively, could connect concepts within the same paper
        # For now, relations are a placeholder
        # TODO: If arXiv has a structured taxonomy, populate accordingly

        # Save cache
        data_to_cache = {
            'documents': documents,
            'concepts': concepts,
            'relations': relations,
            'annotations': dict(doc_annotations)
        }
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_cache, f)

        return Dataset(documents, concepts, relations, dict(doc_annotations))

    def _build_category_graph_bfs(self, root_category: str, max_depth: int):
        """
        Perform BFS traversal on category graph starting from root_category.
        Return list of edges (parent, child).
        Since Wikipedia API does not provide graph directly, use MediaWiki API to query category hierarchy.
        """
        base_url = "https://en.wikipedia.org/w/api.php"
        visited = set()
        queue = deque()
        category_graph = []
        category_to_id = {}
        id_to_category = {}
        category_id_counter = 0

        def get_subcategories(category_title):
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category_title}',
                'cmtype': 'subcat',
                'cmlimit': '500'
            }
            response = requests.get(base_url, params=params).json()
            subcats = response.get('query', {}).get('categorymembers', [])
            return [subcat['title'].replace('Category:', '') for subcat in subcats]

        queue.append((root_category, 0))
        category_to_id[root_category] = category_id_counter
        id_to_category[category_id_counter] = root_category
        category_id_counter += 1

        while queue:
            current_cat, depth = queue.popleft()
            if depth >= max_depth:
                continue
            subcategories = get_subcategories(current_cat)
            for subcat in subcategories:
                if subcat not in category_to_id:
                    category_to_id[subcat] = category_id_counter
                    id_to_category[category_id_counter] = subcat
                    category_id_counter += 1
                parent_id = category_to_id[current_cat]
                child_id = category_to_id[subcat]
                category_graph.append((parent_id, child_id))
                queue.append((subcat, depth + 1))
        return category_graph, category_to_id, id_to_category

    def _get_category_pages(self, concept_to_id: Dict[str, int], max_pages: int = 5000):
        """
        Given category IDs, retrieve pages (titles + summaries) belonging to each category.
        Limit to max_pages per category.
        """
        base_url = "https://en.wikipedia.org/w/api.php"
        results = defaultdict(list)
        for category, cat_id in concept_to_id.items():
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmlimit': max_pages,
                'cmtype': 'page'
            }
            response = requests.get(base_url, params=params).json()
            pages = response.get('query', {}).get('categorymembers', [])
            for page in pages:
                page_title = page['title']
                summary = self._get_page_summary(page_title)
                results[cat_id].append({'title': page_title, 'summary': summary})
        return results

    def _get_page_summary(self, page_title: str) -> str:
        """
        Retrieve the summary (extracted as text before first section).
        """
        api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/{}".format(page_title.replace(' ', '_'))
        try:
            resp = requests.get(api_url).json()
            summary = resp.get('extract', '')
            return summary
        except:
            return ""

    def _combine_title_summary(self, title: str, summary: str) -> str:
        """
        Concatenate title and summary to form document text.
        """
        return f"Title: {title}\nSummary: {summary}"

    def _extract_concepts_from_categories(self, categories_str: str) -> List[str]:
        """
        Parse categories string, e.g., primary categories or keywords, into concepts.
        Implement heuristic: split by delimiters, clean.
        """
        concepts = []
        if not categories_str:
            return concepts
        # Example heuristic: split by semicolons or spaces
        for cat in categories_str.split(';'):
            cat_clean = cat.strip()
            if cat_clean:
                concepts.append(cat_clean)
        return concepts
```

## evaluation.py

```python
## evaluation.py
"""
This module provides functions to evaluate the similarity between generated
and ground-truth ontologies using advanced embedding-based metrics and motif
distribution distance. It relies on pretrained sentence transformers for semantic
representations, the Hungarian algorithm for optimal bipartite matching, and motif
counts for structural comparison.

Functions:
- graph_f1_score: computes combined node and edge F1 score based on embedding similarities.
- motif_distance: measures structural difference using 3-node motif distributions.
- semantic_similarity: computes cosine similarity between concept labels using sentence-transformers.
- node_embedding: computes and caches embeddings for concept labels.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from collections import Counter
from itertools import combinations
import networkx as nx
from typing import Dict, Set, Tuple
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    # If sentence-transformers is not installed, embedder will be None.
    SentenceTransformer = None

# Load the sentence embedding model globally for efficiency.
# Using the model specified in the config, e.g., "all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = None

def _load_embedder():
    global embedder
    if embedder is None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers package is required for embedding-based metrics."
            )
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return embedder

def semantic_similarity(label1: str, label2: str) -> float:
    """
    Compute cosine similarity between two concept labels using sentence embeddings.
    Returns similarity in [-1, 1], higher indicates closer meaning.
    """
    embed = _load_embedder()
    emb1 = embed.encode(label1, convert_to_numpy=True, normalize_embeddings=True)
    emb2 = embed.encode(label2, convert_to_numpy=True, normalize_embeddings=True)
    sim = 1 - cosine(emb1, emb2)
    # Ensure similarity in [-1,1]
    return np.clip(sim, -1, 1)

def node_embedding(concept_label: str, cache: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Generate or retrieve cached embedding for a concept label.
    """
    if concept_label in cache:
        return cache[concept_label]
    embed = _load_embedder()
    emb = embed.encode(concept_label, convert_to_numpy=True, normalize_embeddings=True)
    cache[concept_label] = emb
    return emb

def graph_f1_score(
    model_graph,
    true_graph,
    semantic_threshold: float = 0.436
) -> float:
    """
    Compute the combined F1 score between generated and true graphs using embeddings.
    The matchings for nodes and edges are determined based on maximum cosine similarity
    with a threshold to consider as a valid match.
    """
    # Create embedding caches
    node_cache_true: Dict[str, np.ndarray] = {}
    node_cache_pred: Dict[str, np.ndarray] = {}

    # Prepare node labels
    true_nodes = list(true_graph.nodes)
    pred_nodes = list(model_graph.nodes)

    # Compute node similarity matrix
    true_embs = np.array([node_embedding(lbl, node_cache_true) for lbl in true_nodes])
    pred_embs = np.array([node_embedding(lbl, node_cache_pred) for lbl in pred_nodes])
    sim_matrix_nodes = np.matmul(true_embs, pred_embs.T)  # shape: (true_n, pred_n)

    # Convert similarity to cost for Hungarian (maximize similarity)
    # Use -sim as cost
    cost_matrix_nodes = -sim_matrix_nodes
    true_idx, pred_idx = linear_sum_assignment(cost_matrix_nodes)

    matched_node_pairs = []
    total_node_sim = 0.0
    for i, j in zip(true_idx, pred_idx):
        sim = sim_matrix_nodes[i, j]
        if sim >= semantic_threshold:
            total_node_sim += sim
            matched_node_pairs.append((true_nodes[i], pred_nodes[j], sim))
    # Calculate node precision and recall
    precision_nodes = len(matched_node_pairs) / max(len(pred_nodes), 1)
    recall_nodes = len(matched_node_pairs) / max(len(true_nodes), 1)
    # Harmonic mean
    if precision_nodes + recall_nodes == 0:
        node_f1 = 0.0
    else:
        node_f1 = 2 * precision_nodes * recall_nodes / (precision_nodes + recall_nodes)

    # Now, match edges
    # For each predicted edge, find the best matching true edge based on endpoint embeddings
    true_edges = list(true_graph.edges)
    pred_edges = list(model_graph.edges)
    if len(pred_edges) == 0 or len(true_edges) == 0:
        edge_f1 = 0.0
    else:
        # Create matrices of source and target embeddings for true and predicted edges
        true_edge_embs = []
        for u, v in true_edges:
            u_emb = node_embedding(u, node_cache_true)
            v_emb = node_embedding(v, node_cache_true)
            true_edge_embs.append((u_emb, v_emb))
        pred_edge_embs = []
        for u, v in pred_edges:
            u_emb = node_embedding(u, node_cache_pred)
            v_emb = node_embedding(v, node_cache_pred)
            pred_edge_embs.append((u_emb, v_emb))
        # Form cost matrix for edges
        cost_matrix_edges = np.zeros((len(true_edges), len(pred_edges)))
        for i, (u_true_emb, v_true_emb) in enumerate(true_edge_embs):
            for j, (u_pred_emb, v_pred_emb) in enumerate(pred_edge_embs):
                # Similarity between edges via source and target embeddings
                # For example: average of source similarity and target similarity
                u_sim = 1 - cosine(u_true_emb, u_pred_emb)
                v_sim = 1 - cosine(v_true_emb, v_pred_emb)
                cost = -((u_sim + v_sim) / 2)  # maximize average similarity
                cost_matrix_edges[i, j] = cost
        # Hungarian matching
        true_idx_e, pred_idx_e = linear_sum_assignment(cost_matrix_edges)
        total_edge_sim = 0.0
        for i, j in zip(true_idx_e, pred_idx_e):
            sim = -cost_matrix_edges[i, j]
            if sim >= semantic_threshold:
                total_edge_sim += sim
        # Compute precision and recall for edges
        precision_edges = len(true_idx_e) / max(len(pred_edges), 1)
        recall_edges = len(true_idx_e) / max(len(true_edges), 1)
        if precision_edges + recall_edges == 0:
            edge_f1 = 0.0
        else:
            edge_f1 = 2 * precision_edges * recall_edges / (precision_edges + recall_edges)

    # Combine node and edge scores (simple harmonic mean)
    if node_f1 + edge_f1 == 0:
        combined_f1 = 0.0
    else:
        combined_f1 = 2 * node_f1 * edge_f1 / (node_f1 + edge_f1)

    return combined_f1

def motif_distance(
    true_graph,
    model_graph,
    motif_k: int = 3
) -> float:
    """
    Compute the total variation distance between motif distributions in the two graphs.
    Counts all subgraphs of size motif_k and compares their distributions.
    """
    def count_3node_motifs(g: nx.DiGraph) -> Dict[str, int]:
        """
        Counts specific types of 3-node motifs in graph g, returns counts as a dict.
        For simplicity, counts:
        - chain (A->B->C)
        - cycle (A->B->C->A)
        - feedforward (A->B, A->C, B->C)
        """
        motif_counts = Counter()
        nodes = list(g.nodes)
        total_triplets = 0
        for triplet in combinations(nodes, 3):
            subg = g.subgraph(triplet)
            total_triplets += 1
            # Determine motif type
            # For simplicity, consider only presence of edges
            edges = list(subg.edges())
            # Flags
            has_ab = subg.has_edge(triplet[0], triplet[1])
            has_bc = subg.has_edge(triplet[1], triplet[2])
            has_ca = subg.has_edge(triplet[2], triplet[0])
            has_ac = subg.has_edge(triplet[0], triplet[2])
            # Count patterns
            if has_ab and has_bc and not has_ca:
                motif_counts['chain'] += 1
            elif has_ab and has_bc and has_ca:
                motif_counts['cycle'] += 1
            elif has_ab and has_ac and (not has_bc):
                # feedforward example
                motif_counts['feedforward'] += 1
            # Additional motifs can be added as needed.
        # Normalize counts
        total = total_triplets if total_triplets > 0 else 1
        for key in motif_counts:
            motif_counts[key] /= total
        return dict(motif_counts)

    true_counts = count_3node_motifs(true_graph)
    pred_counts = count_3node_motifs(model_graph)

    # Combine all motif types
    all_motifs = set(true_counts.keys()).union(set(pred_counts.keys()))
    total_var = 0.0
    for motif in all_motifs:
        ptrue = true_counts.get(motif, 0.0)
        ppred = pred_counts.get(motif, 0.0)
        total_var += abs(ptrue - ppred)
    tv_distance = 0.5 * total_var
    return tv_distance

# Additional functions for detailed visualization, result reporting, etc.,
# can be added as needed, but are outside the scope of this core implementation.
```

## graph_utils.py

```python
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
```

## hyperparameter_search.py

```python
## hyperparameter_search.py
"""
This script performs systematic grid search to optimize key hyperparameters
for the End-to-End Ontology Learning pipeline as specified in the paper.
It varies parameters such as alpha, beta thresholds for pruning, and the average relation count M,
evaluates the resulting ontology quality on validation data using the Graph F1 metric,
and identifies the best hyperparameter combination.

It interfaces with existing modules:
- dataset_loader.DatasetLoader to load validation dataset
- model.BingModel for subgraph generation
- evaluation.graph_f1_score to evaluate ontology quality
- graph_utils for graph manipulations
"""

import itertools
import logging
from typing import Dict, Tuple
import yaml
import json
import numpy as np

# Import relevant modules from the project
from dataset_loader import DatasetLoader
from model import BingModel
from evaluation import graph_f1_score
from graph_utils import Graph, remove_cycles

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Extract validation dataset parameters and model configs
dataset_name = config['training'].get('dataset', 'wikipedia')  # fallback to default
model_name = config['model'].get('base_model_name', 'mistral-7b-v0.2')
use_lora = config['model'].get('use_lora', True)
lora_rank = config['model'].get('lora_rank', 32)
finetune_dataset = config['model'].get('finetune_on_dataset', 'wikipedia')

# Hyperparameter grid (values for testing, can be extended)
alpha_candidates = np.linspace(0.1, 1.0, 5).tolist()  # e.g., [0.1, 0.325, 0.55, 0.775, 1.0]
beta_candidates = np.linspace(0.1, 1.0, 5).tolist()
M_candidates = [50, 100, 150, 200]  # average relation occurrence numbers

# Also consider other parameters like relation path length if needed
path_length_candidates = [3, 4]  # as per the datasets

# Load validation dataset
loader = DatasetLoader(config)
validation_dataset = loader.load_wikipedia() if dataset_name == 'wikipedia' else loader.load_arxiv()

# Prepare the model
model = BingModel(
    model_name=model_name,
    use_lora=use_lora,
    lora_rank=lora_rank,
    finetune_on_dataset=finetune_dataset
)

# Function to generate ontology graph for a given document set with current hyperparameters
def generate_full_ontology(
    model: "BingModel",
    dataset: "Dataset",
    alpha: float,
    beta: float,
    M: float,
    path_length: int
) -> Graph:
    """
    Generates and aggregates subgraphs for all documents in dataset
    using the current hyperparameters, applies post-processing, and returns
    the final pruned, cycle-removed ontology graph.
    """
    import copy

    # Initialize an empty directed graph
    full_graph = Graph()
    all_edges = dict()  # key: (concept1, concept2), value: accumulated weight

    # Generate subgraph for each document
    for idx, doc in enumerate(dataset.documents):
        # Consider concepts as context
        concepts = dataset.annotations.get(idx, [])
        # Generate prompt - could be a summarized or concatenated prompt template
        prompt = f"Document: {doc}\nConcepts: {', '.join(concepts)}\nRelation paths:"
        generated_text = model.generate_subgraph(prompt, max_tokens=512, sampling_params={'temperature':0.1, 'top_p':0.9})

        # Parse the generated text into relation paths (implement as needed)
        relations = parse_relations_from_text(generated_text)
        # relations: list of tuples (concept1, relation_type, concept2)
        for u, rel_type, v in relations:
            key = (u, v)
            # Accumulate weights based on frequency
            all_edges[key] = all_edges.get(key, 0) + 1

    # Create graph from accumulated edges
    for (u, v), weight in all_edges.items():
        full_graph.add_edge(u, v, weight=weight)

    # Post-processing: prune edges based on thresholds
    # 1. Absolute threshold (alpha)
    all_weights = np.array([d['weight'] for u, v, d in full_graph.edges(data=True)])
    if len(all_weights) == 0:
        # no edges, return empty graph
        return full_graph
    alpha_thresh_value = np.quantile(all_weights, alpha)
    edges_to_keep = [(u, v) for u, v, d in full_graph.edges(data=True) if d['weight'] >= alpha_thresh_value]
    full_graph = full_graph.edge_subgraph(edges_to_keep).copy()

    # 2. Relative threshold (beta) per node
    # For each node, keep only top edges covering beta cumulative weight
    for u in list(full_graph.nodes):
        out_edges = list(full_graph.out_edges(u, data=True))
        if not out_edges:
            continue
        total_w = sum(d['weight'] for _, _, d in out_edges)
        out_edges_sorted = sorted(out_edges, key=lambda x: x[2]['weight'])
        cumsum = 0.0
        edges_to_remove_from_u = []
        for _, v, d in out_edges_sorted:
            cumsum += d['weight']
            if total_w > 0:
                ratio = cumsum / total_w
            else:
                ratio = 0
            if ratio > beta:
                # Remove edges beyond this point
                for _, v2, d2 in out_edges_sorted[out_edges_sorted.index((_, v, d))+1:]:
                    edges_to_remove_from_u.append((u, v2))
        for edge in edges_to_remove_from_u:
            if full_graph.has_edge(*edge):
                full_graph.remove_edge(*edge)

    # 3. Remove self-loops
    nx_edges = list(full_graph.edges())
    for u, v in nx_edges:
        if u == v:
            full_graph.remove_edge(u, v)

    # 4. Remove cycles (greedy)
    full_graph = remove_cycles(full_graph, strategy='greedy')

    return full_graph

def parse_relations_from_text(text: str) -> list:
    """
    Implement parser to extract relations in (concept1, relation_type, concept2)
    from generated text, based on expected schema or pattern matching.
    """
    relations = []
    # As exact parser details are not specified, implement a simple heuristic
    # E.g., look for lines starting with '-' and contain '->'
    import re
    pattern = re.compile(r'-\s*(.+?)\s*->\s*(.+?)\s*->\s*(.+)')
    lines = text.splitlines()
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            u, rel_type, v = match.groups()
            u = u.strip()
            v = v.strip()
            rel_type = rel_type.strip()
            relations.append((u, rel_type, v))
    return relations


# Grid search over hyperparameters
results = []

for alpha, beta, M in itertools.product(alpha_candidates, beta_candidates, M_candidates):
    logging.info(f"Testing alpha={alpha:.3f}, beta={beta:.3f}, M={M}")
    # Generate ontology with current hyperparameters
    ontology_graph = generate_full_ontology(model, validation_dataset, alpha, beta, M, path_length=4)

    # Evaluate on validation set
    # Provide true ground-truth graph for comparison
    true_graph = Graph()
    # Assume dataset contains groundtruth relations, build the true graph
    for u, v, rel_type in validation_dataset.relations:
        true_graph.add_edge(u, v, weight=1.0)

    # Compute metric
    score = graph_f1_score(ontology_graph, true_graph)
    results.append({
        'alpha': alpha,
        'beta': beta,
        'M': M,
        'validation_score': score
    })
    logging.info(f"Validation Graph F1 score: {score:.4f}")

# Find best hyperparameters
best_result = max(results, key=lambda x: x['validation_score'])
print(f"Best hyperparameters found: alpha={best_result['alpha']:.3f}, beta={best_result['beta']:.3f}, M={best_result['M']}")
print(f"Validation Graph F1: {best_result['validation_score']:.4f}")

# Save or return best hyperparameters
best_params = {
    'alpha': best_result['alpha'],
    'beta': best_result['beta'],
    'M': best_result['M']
}

# (Optional) Save results for analysis
with open("hyperparameter_results.json", "w") as f:
    json.dump(results, f, indent=4)
```

## main.py

```python
## main.py
"""
Main script to run the end-to-end ontology learning pipeline as described in the paper.
It loads configs, datasets, performs hyperparameter tuning, finetunes the model,
generates subgraphs for each document, aggregates and prunes to produce final ontology,
evaluates using custom metrics, and visualizes results.
"""

import yaml
import os
import random
import numpy as np

from dataset_loader import DatasetLoader
from model import BingModel
from prompt_generator import get_chain_of_thought_prompt, get_direct_prompt
from graph_utils import Graph, remove_cycles
from evaluation import graph_f1_score, motif_distance
from visualization import visualize_graph
import hyperparameter_search

# --------------------------------------------
# 1. Load configuration
# --------------------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set random seed for reproducibility
SEED = config.get('misc', {}).get('seed', 42)
random.seed(SEED)
np.random.seed(SEED)

# --------------------------------------------
# 2. Load datasets
# --------------------------------------------
loader = DatasetLoader(config)
dataset_name = config.get('training', {}).get('dataset', 'wikipedia')
if dataset_name == 'wikipedia':
    full_dataset = loader.load_wikipedia()
elif dataset_name == 'arxiv':
    full_dataset = loader.load_arxiv()
else:
    raise ValueError(f"Unknown dataset: {dataset_name}")

# 3. Split dataset into train/validation/test
# Based on the protocol: partition top-level concepts, then within max depth
# For simplicity, assume the datasets are already structured as needed or use provided split functions.
# If not, implement your split following the description, here we proceed with full_dataset as is.

# 4. Hyperparameter tuning (grid search)
# ------------------------------------------------
# Define grid of hyperparameters for pruning thresholds and M (relation avg count)
alpha_grid = np.geomspace(0.1, 1.0, num=5)  # e.g., [0.1, 0.22, 0.45, 0.72, 1.0]
beta_grid = np.geomspace(0.1, 1.0, num=5)   # same as above
M_candidates = [50, 100, 150, 200]
path_length_candidates = [3, 4]

# Run hyperparameter search
best_params = hyperparameter_search.search(
    dataset=full_dataset,
    model_name=config['model']['base_model_name'],
    use_lora=config['model']['use_lora'],
    lora_rank=config['model']['lora_rank'],
    dataset_type=dataset_name,
    alpha_grid=alpha_grid,
    beta_grid=beta_grid,
    M_candidates=M_candidates,
    path_length_candidates=path_length_candidates,
    validation_function=graph_f1_score,
    # specify other needed configs
    # e.g that hyperparameter_search.py uses
)

# After search, best_params contains {'alpha':..., 'beta':..., 'M':..., 'path_length':...}
alpha_opt = float(best_params['alpha'])
beta_opt = float(best_params['beta'])
M_opt = int(best_params['M'])
path_length = int(best_params['path_length'])

# --------------------------------------------
# 5. Initialize and fine-tune the model
# --------------------------------------------
model = BingModel(
    model_name=config['model']['base_model_name'],
    use_lora=config['model']['use_lora'],
    lora_rank=config['model']['lora_rank']
)

# Fine-tune on full dataset annotations (for Wikipedia) or transfer on small arXiv data
model.finetune(full_dataset, epochs=2, loss_masking=True)

# For transfer to arXiv, perform further fine-tuning if specified
if dataset_name == 'arxiv' and 'finetune_on_dataset' in config['model']:
    # Load arXiv samples for transfer fine-tuning
    # To simulate, reuse full_dataset or load a small subset
    # Here, assume 'full_dataset' is already arXiv data or load accordingly
    model.finetune(full_dataset, epochs=1, loss_masking=True)  # or fine-tune on small set

# --------------------------------------------
# 6. Generate subgraphs for each document
# --------------------------------------------
generated_subgraphs = []
for idx, document in enumerate(full_dataset.documents):
    # Prepare prompt for subgraph generation
    concepts = full_dataset.annotations.get(idx, [])
    prompt_text = get_chain_of_thought_prompt(document, concepts)
    # Generate subgraph text
    gen_text = model.generate_subgraph(
        prompt_text,
        max_tokens=512,
        sampling_params={'temperature': 0.1, 'top_p': 0.9}
    )
    # Parse generated text into relations
    relations = []
    import re
    pattern = re.compile(r'-\s*(.+?)\s*->\s*(.+?)\s*->\s*(.+)')
    for line in gen_text.splitlines():
        match = pattern.match(line.strip())
        if match:
            u, rel, v = match.groups()
            u, v, rel = u.strip(), v.strip(), rel.strip()
            relations.append((u, rel, v))
    # Save subgraph (as Graph object or similar)
    g = Graph()
    for u, rel, v in relations:
        g.add_edge(u, v, weight=1.0)  # initial weight 1 for each occurrence
    generated_subgraphs.append(g)

# --------------------------------------------
# 7. Aggregate all subgraphs
# --------------------------------------------
# Combine all subgraphs by summing edge weights
global_graph = Graph()
for g in generated_subgraphs:
    for u, v, d in g.edges(data=True):
        if global_graph.has_edge(u, v):
            global_graph[u][v]['weight'] += 1
        else:
            global_graph.add_edge(u, v, weight=1.0)

# --------------------------------------------
# 8. Post-processing (pruning and cycle removal)
# --------------------------------------------
# Prune edges based on hyperparameters
# Absolute thresholding by alpha
all_weights = np.array([d['weight'] for u, v, d in global_graph.edges(data=True)])
if len(all_weights) > 0:
    alpha_thresh = np.quantile(all_weights, alpha_opt)
    edges_to_keep = [(u, v) for u, v, d in global_graph.edges(data=True) if d['weight'] >= alpha_thresh]
    global_graph = global_graph.edge_subgraph(edges_to_keep).copy()

# Remove inverse edges (keep only one direction based on weight)
from graph_utils import remove_inverse_edges
global_graph = remove_inverse_edges(global_graph)

# Remove self-loops
from graph_utils import remove_self_loops
global_graph = remove_self_loops(global_graph)

# Remove cycles greedily if configured
if config.get('post_processing', {}).get('remove_cycles', True):
    global_graph = remove_cycles(global_graph, strategy='greedy')

# Remove nodes with no edges
import networkx as nx
isolated_nodes = [n for n, deg in global_graph.degree() if deg == 0]
global_graph.remove_nodes_from(isolated_nodes)

# --------------------------------------------
# 9. Evaluate the final ontology
# --------------------------------------------
# Load ground truth ontology for comparison
# Assume ground truth is in full_dataset.relations as list
true_graph = Graph()
if hasattr(full_dataset, 'relations'):
    for u, rel, v in full_dataset.relations:
        true_graph.add_edge(u, v, weight=1.0)
else:
    # No ground truth
    true_graph = None

# Compute evaluation metrics
graph_f1 = graph_f1_score(global_graph, true_graph)
motif_dist = motif_distance(true_graph, global_graph, motif_k=3)

# 10. Visualization
# --------------------------------------------
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Visualize the final ontology graph
visualize_graph(global_graph, os.path.join(output_dir, "final_ontology.png"))

# Optional: visualize subgraph around key concepts
# For illustration, pick a random node
if len(global_graph.nodes) > 0:
    sample_node = list(global_graph.nodes)[0]
    visualize_graph(
        global_graph,
        os.path.join(output_dir, "subgraph_sample.png"),
        title=f"Subgraph around {sample_node}"
    )

# --------------------------------------------
# 11. Save results
# --------------------------------------------
# Save the ontology graph to file (GraphML)
global_graph.save_graph(os.path.join(output_dir, "ontology.graphml"))

# Save evaluation metrics
with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w") as f:
    f.write(f"Graph F1: {graph_f1:.4f}\n")
    f.write(f"Motif Distance: {motif_dist:.4f}\n")
    # Add more metrics as needed

print("Ontology learning pipeline completed.")
print(f"Graph F1: {graph_f1:.4f}")
print(f"Motif Distance: {motif_dist:.4f}")
```

**Notes:**

- This code presumes the existence of the other modules (`dataset_loader.py`, `model.py`, etc.) with their defined classes and functions.
- Parsing the model output into relations (`parse_relations_from_text`) should be adapted to match your actual model output format.
- The hyperparameter tuning via `hyperparameter_search.search()` is invoked with the needed parameters, assuming it takes dataset, model parameters, validation function, and hyperparameter grids.
- Final saving and visualization paths are arranged sensibly, and the pipeline's flow follows the described methodology, ensuring reproducibility.
- Placeholder assumptions (e.g., ground-truth relations, exact prompt templates) should be replaced with actual implementation detail as per your environment and data.

## model.py

```python
## model.py

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Tuple
import random
import numpy as np

# If using LoRA, import from peft
try:
    from peft import get_peft_model, LoraConfig
except ImportError:
    # In case peft is not installed, define dummy functions/classes
    def get_peft_model(model, peft_config):
        return model

    class LoraConfig:
        def __init__(self, r, lora_alpha, lora_dropout=0.0, target_modules=None):
            pass

# Loader for device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BingModel:
    def __init__(self, model_name: str = "mistral-7b-v0.2",
                 use_lora: bool = True,
                 lora_rank: int = 32,
                 finetune_on_dataset: str = "wikipedia",
                 loss_masking: bool = True,
                 relation_masking_M: int = 100,
                 mask_mask_prob: float = 0.5,
                 gradient_clipping_norm: float = 1.0):
        """
        Initialize the Large Language Model with optional LoRA modules.
        """
        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.finetune_on_dataset = finetune_on_dataset
        self.loss_masking = loss_masking
        self.relation_masking_M = relation_masking_M
        self.mask_mask_prob = mask_mask_prob
        self.gradient_clipping_norm = gradient_clipping_norm

        # Load pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        self.model.to(device)

        # Initialize LoRA if required
        if self.use_lora:
            peft_config = LoraConfig(r=self.lora_rank, lora_alpha=16, target_modules=["layers.*.attn", "layers.*.feed_forward"])
            self.model = get_peft_model(self.model, peft_config)
            # Note: If 'peft' not installed, this step skips or would need adjustment.

        # Set model to train/eval modes accordingly
        self.model.train()

    def finetune(self, train_data, epochs: int = 2, loss_masking: bool = True):
        """
        Fine-tune the model with optional custom masked loss regularizer.
        train_data: object with attributes: documents, concepts, relations, annotations
        """
        from torch.utils.data import DataLoader

        # Prepare dataset: each item is (prompt, target_text)
        dataset = []

        for idx, doc in enumerate(train_data.documents):
            concepts = train_data.annotations.get(idx, [])
            prompt = self._construct_training_prompt(doc, concepts)
            target_sequence = self._construct_training_target(concepts)
            dataset.append((prompt, target_sequence, concepts))

        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=self._collate_fn)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)  # hyperparameter tuned
        best_val_score = -float('inf')
        patience = 2
        patience_counter = 0

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                input_ids, attention_mask, labels, mask_flags, relation_counts = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                self.model.train()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # If loss_masking enabled, apply relation-based masking regularizer
                if self.loss_masking:
                    # Apply custom masking: for relations in relation_counts dictionary
                    loss = self._apply_loss_mask(loss, mask_flags, relation_counts)

                optimizer.zero_grad()
                loss.backward()

                if self.gradient_clipping_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping_norm)

                optimizer.step()
                total_loss += loss.item()

            # Optional: validation step here, compute validation metrics, save best model
            # Skipping for brevity; could add validation code and early stopping based on validation metric.

    def generate_subgraph(self, prompt: str, max_tokens: int = 512, sampling_params: Dict = None) -> str:
        """
        Generate a subgraph string from prompt with specified sampling parameters.
        """
        if sampling_params is None:
            sampling_params = {'temperature': 0.1, 'top_p': 0.9}

        encoded = self.tokenizer(prompt, return_tensors='pt').to(device)
        generate_kwargs = dict(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=sampling_params.get('temperature', 0.1),
            top_p=sampling_params.get('top_p', 0.9),
            num_return_sequences=1
        )

        with torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text

    def _construct_training_prompt(self, document: str, concepts: list) -> str:
        """
        Create a training prompt with document text and concept list.
        Use the template matching the paper's appendix if available.
        """
        prompt_template = (
            "Given the document:\n"
            "{doc}\n"
            "and the list of concepts:\n"
            "{concepts}\n"
            "Generate the relevant concept relation subgraph as a list of paths in the form:\n"
            "- {concept} -> {relation} -> {concept}\n"
        )
        return prompt_template.format(doc=document, concepts=', '.join(concepts))

    def _construct_training_target(self, concepts: list) -> str:
        """
        Construct the target sequence for training.
        """
        # For simplicity, a placeholder; in practice, generate structured paths.
        # For now, assume target is empty, actual parsing needs implementation.
        return ""

    def _collate_fn(self, batch):
        """
        Collate batch with masking considerations if needed.
        """
        # batch: list of tuples (prompt, target_sequence, concepts)
        prompts = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        # Tokenize all prompts
        inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
        labels = self.tokenizer(targets, padding=True, truncation=True, return_tensors='pt')

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        label_ids = labels['input_ids']

        # Generate mask flags and relation counts for masking loss
        # For simplicity, generate dummy mask_flags and relation_counts
        batch_size = input_ids.shape[0]
        mask_flags = torch.zeros_like(label_ids, dtype=torch.bool)
        relation_counts = {}  # e.g., {relation_path: count}

        # In real implementation, parse target sequences to find relation tokens and compute their counts

        return input_ids, attention_mask, label_ids, mask_flags, relation_counts

    def _apply_loss_mask(self, loss, mask_flags, relation_counts):
        """
        Apply the custom masked loss based on relation frequency.
        """
        # For simplicity, assume uniform masking
        # In practice, analyze relation_counts and mask proportionally
        # We can implement a stochastic masking based on relation frequency
        # But as per the constraints, provide a placeholder
        return loss

    def _parse_generated_text(self, generated_text: str):
        """
        Parse the generated text to extract concept relations/paths.
        This can be implemented via regex matching according to promotion template.
        """
        # Placeholder: implement parser to extract relation triplets or paths
        relations = []
        #...
        return relations
```

## prompt_generator.py

```python
## prompt_generator.py

"""
This module provides functions to generate prompts for training and inference
of the ontology learning model, following the styles specified in the paper.
It uses prompt templates defined based on the paper's appendix figures and description.
The functions include:
- get_chain_of_thought_prompt()
- get_direct_prompt()
- get_instruction_prompt()

They are designed to ensure prompt consistency and fidelity to the original paper's
prompt styles, facilitating reliable subgraph generation and reproducibility.
"""

from typing import List, Dict

# Define default prompt templates matching figures 6-8 from appendix
# These templates incorporate placeholders for document text, concept list, and reasoning steps.

def get_chain_of_thought_prompt(
    document_text: str,
    concepts: List[str],
    additional_parameters: Dict = None
) -> str:
    """
    Generates a chain-of-thought prompt for model to generate concept subgraphs with reasoning.
    This prompt guides the model through explicit reasoning steps.

    Args:
        document_text (str): The text of the document (e.g., summary, abstract).
        concepts (List[str]): List of concepts relevant to the document.
        additional_parameters (Dict, optional): Additional parameters for prompt styling.

    Returns:
        str: The formatted chain-of-thought prompt string.
    """
    # Retrieve additional parameters if provided
    params = additional_parameters if additional_parameters else {}

    # Use the predefined template or define a default one
    template = params.get(
        "template",
        """Given the document:
{doc}
and the list of concepts:
{concepts}

Explain your reasoning step by step. Based on your reasoning, 
list the relevant concept relation paths as a list of sequences where each sequence is a chain of concepts connected by '->'. 
For example:
- {concept1} -> {relation} -> {concept2} -> {relation} -> {concept3}
After reasoning, generate the concept subgraph in the form of paths listed above, each on a new line.

Please elucidate your reasoning clearly and produce the relation paths accordingly."""
    )

    # Format the prompt
    prompt = template.format(
        doc=document_text,
        concepts=', '.join(concepts)
    )
    return prompt


def get_direct_prompt(
    document_text: str,
    concepts: List[str],
    additional_parameters: Dict = None
) -> str:
    """
    Generates a direct, instruction-style prompt for the model to produce relevant concept relations,
    without explicit reasoning steps, suitable for zero-shot or inference.

    Args:
        document_text (str): The text of the document.
        concepts (List[str]): List of concepts relevant to the document.
        additional_parameters (Dict, optional): Additional parameters.

    Returns:
        str: The formatted direct prompt string.
    """
    params = additional_parameters if additional_parameters else {}

    template = params.get(
        "template",
        """Given the document:
{doc}
and the list of concepts:
{concepts}
Provide the concept relation subgraph associated with this document as a list of relation paths, each on a new line, in the form:
- {concept} -> {relation} -> {concept}
List only the relevant relations inferred from the document. Do not include explanations or reasoning."""
    )

    prompt = template.format(
        doc=document_text,
        concepts=', '.join(concepts)
    )
    return prompt


def get_instruction_prompt(
    instruction_type: str = "task_instructions",
    additional_parameters: Dict = None
) -> str:
    """
    Generates a general instruction prompt used as a system prompt or task description.
    This can include dataset details, task description, or formatting guidelines.

    Args:
        instruction_type (str): Type of instruction prompt, e.g., "task_instructions".
        additional_parameters (Dict, optional): Additional context or instructions.

    Returns:
        str: The instruction prompt string.
    """
    # For simplicity, we define a static template matching the paper style.
    # This can be extended based on 'instruction_type' or other parameters.
    template = additional_parameters.get(
        "template",
        """You are an AI assistant tasked with constructing part of an ontology. 
Given documents and concepts, generate subgraphs representing their hierarchical relations.
Use clear, concise language and produce relation paths in the format:
- {concept} -> {relation} -> {concept}
Ensure outputs are well-structured and adhere to the formatting examples provided in the documentation."""
    )

    return template
```


## trainer.py

```python
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
```

## visualization.py

```python
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
```


---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\ollm\ollm_repo`
