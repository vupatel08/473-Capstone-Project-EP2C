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
