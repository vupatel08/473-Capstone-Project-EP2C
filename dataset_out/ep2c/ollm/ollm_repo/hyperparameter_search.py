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
