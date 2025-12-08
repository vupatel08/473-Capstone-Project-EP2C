## main.py
import yaml
import os
import random
import torch
import numpy as np

from dataset_loader import DatasetLoader
from embedding import Embedding
from retrieval import RetrievalSystem
from pcst_solver import solve_pcst
from graph_processor import GraphProcessor
from prompt_tuning import PromptTuner
from model import LLMModel
from evaluation import Evaluator

def main():
    # Load configuration from YAML
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize DatasetLoader
    dataset_paths = {
        'train': config['dataset']['train_path'],
        'val': config['dataset']['val_path'],
        'test': config['dataset']['test_path']
    }
    dataset_loader = DatasetLoader(dataset_paths)
    train_data, val_data, test_data = dataset_loader.load_data()

    # Initialize text embedding model
    embedding_model = Embedding(
        model_name=config['embedding']['model_name'],
        max_input_tokens=config['embedding']['max_input_tokens'],
        model_precision=config['embedding']['model_precision']
    )

    # Collect all node and edge texts for indexing
    all_node_texts = []
    all_edge_texts = []
    # Store text descriptions for dataset samples
    dataset_samples = []

    # Prepare dataset samples
    for dataset_split in [train_data, val_data, test_data]:
        for data_point in dataset_split:
            text_desc = data_point['text_description']
            dataset_samples.append(data_point)
            # Collect texts for node and edge indexing
            for node in data_point['nodes']:
                txt_node = f"Node: {node['attributes']}"
                all_node_texts.append(txt_node)
            for edge in data_point['edges']:
                src_str = str(edge['src'])
                dst_str = str(edge['dst'])
                txt_edge = f"Edge: {edge['attributes']} from {src_str} to {dst_str}"
                all_edge_texts.append(txt_edge)

    # Encode all node and edge texts
    node_embeddings = embedding_model.encode_nodes(all_node_texts)
    edge_embeddings = embedding_model.encode_edges(all_edge_texts)
    combined_embeddings = np.vstack([node_embeddings, edge_embeddings])

    # Build retrieval index
    retrieval_system = RetrievalSystem(embedding_dim=embedding_model.embedding_dim)
    retrieval_system.build_index(combined_embeddings)

    # Initialize graph encoder (e.g., GAT) for graph embedding
    # (Assuming GraphTransformer in your code)
    # For simplicity, define a placeholder for graph encoder
    from torch import nn
    class GraphEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            # Placeholder for the actual GAT/Transformer model
        def encode_graph(self, graph):
            # Returns a fixed-size graph embedding
            return torch.randn(embedding_model.embedding_dim).to(device)

    graph_encoder = GraphEncoder()

    # Initialize PCST solver with parameters
    edge_cost = config['training'].get('edge_cost', 1.0)
    # Note: Prize assignment handled during retrieval
    # Will pass prizes and solve per example

    # Initialize LLM (frozen or tune-able)
    llm_cfg = {
        'model_name': config['model']['model_name'],
        'model_precision': config['model']['model_precision'],
        'max_input_tokens': config['model']['max_input_tokens']
    }
    llm_model = LLMModel(llm_cfg)
    prompt_tuner = PromptTuner({**llm_cfg,
                                'prompt_length': config['training'].get('prompt_length', 10),
                                'prompt_learning_rate': config['training'].get('prompt_learning_rate', 1e-5),
                                'trainable_method': 'LoRA'})  # or 'prompt_tuning'

    # Set optimizer based on tuning method
    if prompt_tuner.prompt_learning_rate:
        params = []
        if prompt_tuner.trainable_method == 'prompt_tuning':
            params = [prompt_tuner.prompt_tokens]
        elif prompt_tuner.trainable_method == 'LoRA':
            params = filter(lambda p: p.requires_grad, llm_model.model.parameters())
        optimizer = torch.optim.AdamW(params, lr=prompt_tuner.prompt_learning_rate)
    else:
        optimizer = None

    # Initialize evaluator
    evaluator = Evaluator()

    # Define number of epochs
    num_epochs = config['training'].get('epochs', 10)
    batch_size = config['training'].get('batch_size', 16)
    max_input_tokens = config['model']['max_input_tokens']
    retrieval_top_k = config['training'].get('retrieval_top_k', 5)
    prompt_max_new_tokens = 32

    # Helper functions
    def process_example(data_point):
        """
        Process a single data point:
        - Encode question
        - Retrieve relevant nodes/edges
        - Construct subgraph via PCST
        - Textualize subgraph
        - Prepare prompt
        - Generate answer
        - Return generated answer and relevant info
        """
        question_text = data_point['question']
        # Encode question
        z_q = embedding_model.encode_question(question_text)
        # Search for relevant nodes and edges
        indices, dists = retrieval_system.search(z_q, top_k=retrieval_top_k)
        # Assign prizes based on similarity (higher dist -> lower prize)
        # Here, we invert similarity since faiss returns similarity scores; for cosine, higher is better
        # For simplicity, assign prize = (k - rank)
        prizes_nodes = np.zeros(len(all_node_texts))
        prizes_edges = np.zeros(len(all_edge_texts))
        for rank, idx in enumerate(indices):
            prize_value = retrieval_top_k - rank  # higher for top-ranked
            if idx < len(all_node_texts):
                prizes_nodes[idx] = prize_value
            else:
                edge_idx = idx - len(all_node_texts)
                if edge_idx < len(all_edge_texts):
                    prizes_edges[edge_idx] = prize_value

        # Build graph (u,v) with edge prizes
        # Note: in realistic setup, references to true graph node IDs are needed
        # Here, assume a linear chain for demonstration, or use retrieved node IDs
        # For simplicity, assume nodes are numbered 0..N-1 and edges connect sequential nodes
        # Data point contains 'nodes' and 'edges'
        nodes_list = data_point['nodes']
        edges_list = data_point['edges']
        # Build a networkx graph with node attributes and edge attributes
        import networkx as nx
        g = nx.Graph()
        for node in nodes_list:
            node_id = node['node_id']
            g.add_node(node_id, prize=prizes_nodes[node_id], attributes=node['attributes'])
        for idx, edge in enumerate(edges_list):
            src = edge['src']
            dst = edge['dst']
            e_prize = prizes_edges[idx]
            g.add_edge(src, dst, prize=e_prize, attributes=edge['attributes'])

        # Handle virtual nodes and solve PCST
        pruned_subgraph = solve_pcst(
            node_prizes=np.array([d['prize'] for _, d in g.nodes(data=True)]),
            edge_prizes=np.array([d['prize'] for _, _, d in g.edges(data=True)]),
            edge_cost=edge_cost
        )

        # Textualize the subgraph
        graph_dict = {
            'nodes': [{'node_id': n, 'attributes': g.nodes[n]['attributes']} for n in pruned_subgraph.nodes()],
            'edges': [{'src': u, 'dst': v, 'attributes': g.get_edge_data(u, v)} for u, v in pruned_subgraph.edges()]
        }
        textualized_graph = GraphProcessor().textualize_graph(graph_dict)

        # Prepare prompt for generation
        prompt_input = prompt_tuner.prepare_prompt(textualized_graph, question_text)
        input_embeds = prompt_input['inputs_embeds']
        attention_mask = prompt_input['attention_mask']

        # Generate answer
        generated_response = llm_model.generate_from_embeddings(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=prompt_max_new_tokens
        )
        return generated_response, data_point

    # Loop over epochs
    for epoch in range(1, num_epochs + 1):
        print(f"\nStarting epoch {epoch}/{num_epochs}")
        # Shuffle training data
        random.shuffle(train_data)

        total_loss = 0
        num_batches = 0

        # Process batches
        for i in range(0, len(train_data), batch_size):
            batch_samples = train_data[i:i + batch_size]

            # Prepare batch tensors
            batch_input_embeds = []
            batch_attention_mask = []
            batch_labels = []

            optimizer.zero_grad()

            for data_point in batch_samples:
                # Generate model inputs
                generated_text, data_point = process_example(data_point)

                # For training, prepare labels as shifted tokens
                # Here, for simplicity, assume using generated_text as ground truth
                # In real case, use proper tokenization and label shifting
                # Skipping detailed training code for brevity

            # Step entire optimizer after processing batch
            if optimizer:
                optimizer.step()

        # Validation at epoch end
        val_metrics = evaluate(model=llm_model, dataset=val_data, generator_func=lambda dp: process_example(dp)[0])
        val_score = val_metrics.get('accuracy', 0)
        print(f"Epoch {epoch} validation accuracy: {val_score:.4f}")

        # Save best model if improved
        # (Implement checkpoint saving as needed)

if __name__ == "__main__":
    main()
