# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import json
import csv
import random
from typing import List, Dict, Any, Tuple
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, dataset_paths: Dict[str, str], dataset_type: str = None,
                 train_split_ratio: float = 0.6, val_split_ratio: float = 0.2,
                 test_split_ratio: float = 0.2, seed: int = 42):
        """
        Initialize DatasetLoader.
        Args:
            dataset_paths (dict): Dictionary with keys 'train', 'val', 'test' pointing to dataset file paths.
            dataset_type (str): One of 'ExplaGraphs', 'SceneGraphs', 'WebQSP'. If None, inferred from file extension or user.
            train_split_ratio (float): Ratio for training set.
            val_split_ratio (float): Ratio for validation set.
            test_split_ratio (float): Ratio for test set.
            seed (int): Random seed for reproducibility.
        """
        self.dataset_paths = dataset_paths
        self.dataset_type = dataset_type
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
        self.seed = seed
        # Containers for datasets
        self.train_data = []
        self.val_data = []
        self.test_data = []

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load datasets from provided paths, parse and split into train, val, test.
        Returns:
            Tuple of lists: (train_data, val_data, test_data)
        """
        all_data = []

        for split_name, path in self.dataset_paths.items():
            if not os.path.exists(path):
                logger.warning(f"Dataset file {path} not found.")
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext == '.json':
                dataset = self._load_json_dataset(path, dataset_type=self.dataset_type)
            elif ext == '.csv':
                dataset = self._load_csv_dataset(path, dataset_type=self.dataset_type)
            else:
                logger.warning(f"Unknown file extension {ext} for {path}. Skipping.")
                continue

            # Wrap each data point into standardized dict
            for item in dataset:
                graph_data, question, answer, extra = item
                graph_dict = self._convert_triplet_or_structured_data(graph_data, dataset_type=self.dataset_type)
                textual_desc = self.convert_graph_to_text(graph_dict)
                entry = {
                    'graph_id': extra.get('graph_id', 'unknown'),
                    'nodes': graph_dict['nodes'],
                    'edges': graph_dict['edges'],
                    'text_description': textual_desc,
                    'question': question,
                    'answer': answer
                }
                all_data.append(entry)

        # Shuffle data for randomness
        random.seed(self.seed)
        random.shuffle(all_data)

        total = len(all_data)
        train_end = int(total * self.train_split_ratio)
        val_end = train_end + int(total * self.val_split_ratio)

        self.train_data = all_data[:train_end]
        self.val_data = all_data[train_end:val_end]
        self.test_data = all_data[val_end:]

        logger.info(f"Loaded {len(self.train_data)} train, {len(self.val_data)} val, {len(self.test_data)} test samples.")
        return self.train_data, self.val_data, self.test_data

    def _load_json_dataset(self, path: str, dataset_type: str = None) -> List[Tuple]:
        """
        Load and parse dataset from JSON file.
        For each entry, output as tuple: (graph_data, question, answer, extra)
        """
        with open(path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        data_list = []

        # Depending on dataset, parse accordingly
        if 'ExplaGraphs' in path or (dataset_type and dataset_type.lower() == 'expla'):
            # ExplaGraphs: dataset with triplets or explanations
            # Assumed structure: list of dicts with 'triplets' or similar
            for idx, item in enumerate(data_json):
                triplets = item.get('triplets') or item.get('triplet_list') or []
                question = item.get('question', '')
                answer = item.get('label', '')  # support or contradict
                extra = {'graph_id': f"expla_{idx}"}
                data_list.append((triplets, question, answer, extra))
        elif 'SceneGraphs' in path or (dataset_type and dataset_type.lower() == 'sceneg'):
            # SceneGraphs: JSON with object nodes, attributes, relations
            for idx, item in enumerate(data_json):
                graph_structure = item.get('graph', {})  # assume 'graph' key
                question = item.get('question', '')
                answer = item.get('answer', '')
                extra = {'graph_id': f"scene_{idx}"}
                data_list.append((graph_structure, question, answer, extra))
        elif 'WebQSP' in path or (dataset_type and dataset_type.lower() == 'webqsp'):
            # WebQSP: JSON list of question-answer triples
            for idx, item in enumerate(data_json):
                # Assuming item contains 'graph' or 'triplets'
                triplets = item.get('triplets') or []
                question = item.get('question', '')
                answer = item.get('answer', '')
                extra = {'graph_id': f"webqsp_{idx}"}
                data_list.append((triplets, question, answer, extra))
        else:
            # Fallback: try to parse as list of dicts with 'nodes', 'edges'
            for idx, item in enumerate(data_json):
                # Try to interpret as generic graph
                graph_structure = item.get('graph', {})
                question = item.get('question', '')
                answer = item.get('answer', '')
                extra = {'graph_id': f"fallback_{idx}"}
                data_list.append((graph_structure, question, answer, extra))
        return data_list

    def _load_csv_dataset(self, path: str, dataset_type: str = None) -> List[Tuple]:
        """
        Load and parse dataset from CSV.
        """
        data_list = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Expect columns like node_id, node_attr, src, dst, edge_attr etc.
                # For simplicity, treat entire row as graph info
                nodes = []
                edges = []
                # Distinguish node vs edge rows if needed
                # Here, assume nodes info in certain columns
                # For specialized datasets, this can be extended
                node_id = row.get('node_id')
                node_attr = row.get('node_attr')
                if node_id is not None:
                    nodes.append({'node_id': int(node_id), 'attributes': self._parse_attributes(node_attr)})
                src = row.get('src')
                dst = row.get('dst')
                edge_attr = row.get('edge_attr')
                if src is not None and dst is not None:
                    edges.append({'src': int(src), 'dst': int(dst), 'attributes': self._parse_attributes(edge_attr)})
                question = row.get('question', '')
                answer = row.get('answer', '')
                extra = {'graph_id': f"csv_{idx}"}
                graph_struct = {'nodes': nodes, 'edges': edges}
                data_list.append((graph_struct, question, answer, extra))
        return data_list

    def _parse_attributes(self, attr_str: str) -> Dict[str, str]:
        """
        Parse attribute string into dict. E.g., "color: yellow; size: large"
        """
        attributes = {}
        if not attr_str:
            return attributes
        for part in attr_str.split(';'):
            part = part.strip()
            if not part:
                continue
            if ':' in part:
                key, val = part.split(':', 1)
                attributes[key.strip()] = val.strip()
            else:
                attributes[part.strip()] = ''
        return attributes

    def _convert_triplet_or_structured_data(self, data: Any, dataset_type: str = None) -> Dict[str, Any]:
        """
        Convert raw triplet or structured data into standard graph dict
        with 'nodes' and 'edges'.
        """
        nodes_dict = {}
        edges_list = []

        if dataset_type and dataset_type.lower() == 'expla':
            # Data is list of triplets: [(head, relation, tail), ...]
            triplets: List = data
            for triplet in triplets:
                head, relation, tail = triplet
                # Add nodes if not exist
                for n in [head, tail]:
                    if n not in nodes_dict:
                        nodes_dict[n] = {'node_id': len(nodes_dict), 'attributes': {'name': n}}
                src_id = list(nodes_dict.keys()).index(head)
                dst_id = list(nodes_dict.keys()).index(tail)
                edge_attr = {'relation': relation}
                edges_list.append({'src': src_id, 'dst': dst_id, 'attributes': edge_attr})
            nodes_list = [{'node_id': i, 'attributes': nodes_dict[n]['attributes']} for i, n in enumerate(nodes_dict)]
            return {'nodes': nodes_list, 'edges': edges_list}

        elif dataset_type and dataset_type.lower() == 'sceneg':
            # Data is nested JSON with objects, attributes, relations
            # Expect structure with 'objects', 'attributes', 'relations'
            # For simplicity, assume 'nodes' and 'edges'
            objects = data.get('objects', [])
            for idx, obj in enumerate(objects):
                node_attr = obj.get('name', '')
                nodes_dict[idx] = {'node_id': idx, 'attributes': {'name': node_attr}}
            # Build edges from relations
            for rel in data.get('relations', []):
                src_idx = rel.get('src')
                dst_idx = rel.get('dst')
                relation_name = rel.get('relation', '')
                if src_idx is not None and dst_idx is not None:
                    edges_list.append({'src': src_idx, 'dst': dst_idx, 'attributes': {'relation': relation_name}})
            nodes_list = [{'node_id': i, 'attributes': nodes_dict[i]['attributes']} for i in nodes_dict]
            return {'nodes': nodes_list, 'edges': edges_list}

        elif dataset_type and dataset_type.lower() == 'webqsp':
            # Data is list of triplets with entity info
            triplets: List = data
            for trip in triplets:
                head, relation, tail = trip
                if head not in nodes_dict:
                    nodes_dict[head] = {'node_id': len(nodes_dict), 'attributes': {'name': head}}
                if tail not in nodes_dict:
                    nodes_dict[tail] = {'node_id': len(nodes_dict), 'attributes': {'name': tail}}
                src_id = list(nodes_dict.keys()).index(head)
                dst_id = list(nodes_dict.keys()).index(tail)
                edge_attr = {'relation': relation}
                edges_list.append({'src': src_id, 'dst': dst_id, 'attributes': edge_attr})
            nodes_list = [{'node_id': i, 'attributes': nodes_dict[n]['attributes']} for i, n in enumerate(nodes_dict)]
            return {'nodes': nodes_list, 'edges': edges_list}

        else:
            # Default: treat 'data' as structured graph info
            return data

    def convert_graph_to_text(self, graph: Dict[str, Any]) -> str:
        """
        Convert the internal graph representation into a structured natural language description.
        """
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])

        node_strings = []
        for node in nodes:
            node_id = node.get('node_id')
            attrs = node.get('attributes', {})
            attr_str = ', '.join(f"{k}: {v}" for k, v in attrs.items())
            node_strings.append(f"{node_id}: {attr_str}")

        edge_strings = []
        for edge in edges:
            src = edge.get('src')
            dst = edge.get('dst')
            attrs = edge.get('attributes', {})
            relation = attrs.get('relation', 'related to')
            edge_strings.append(f"{src} -> {dst}: {relation}")

        description_parts = []
        if node_strings:
            description_parts.append("Nodes:\n" + "\n".join(node_strings))
        if edge_strings:
            description_parts.append("Edges:\n" + "\n".join(edge_strings))
        # Concatenate with separator
        full_description = "\n".join(description_parts)
        return full_description
```

## embedding.py

```python
## embedding.py
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List
import logging

# Set up logging for debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Embedding:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_input_tokens: int = 512,
                 model_precision: str = "fp16"):
        """
        Initialize the Embedding class with a pretrained text encoder.
        Args:
            model_name (str): Huggingface model identifier.
            max_input_tokens (int): Maximum token length for encoding.
            model_precision (str): 'fp16' or 'fp32' for model weights precision.
        """
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.model_precision = model_precision

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model with specified precision
        self.model = AutoModel.from_pretrained(self.model_name)
        if self.model_precision == "fp16":
            self.model = self.model.half()
        self.model.to(self.device)
        self.model.eval()

        # Determine embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size

        # Freeze model parameters (no training expected)
        for param in self.model.parameters():
            param.requires_grad = False

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into dense vectors.
        Args:
            texts (List[str]): List of input strings.

        Returns:
            np.ndarray: Array of shape (len(texts), embedding_dim)
        """
        # Tokenize with truncation and padding
        encodings = self.tokenizer(
            texts,
            max_length=self.max_input_tokens,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Use CLS token embedding as representation
            # outputs.last_hidden_state shape: [batch_size, seq_len, hidden_size]
            # CLS token is at position 0
            embeddings = outputs.last_hidden_state[:, 0, :]  # shape: [batch_size, hidden_size]
            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
        return embeddings_np

    def encode_nodes(self, nodes: List[str]) -> np.ndarray:
        """
        Encode a list of node attribute strings into vectors.
        Args:
            nodes (List[str]): List of textual node attributes.

        Returns:
            np.ndarray: Embeddings shape (num_nodes, embedding_dim)
        """
        if not nodes:
            return np.zeros((0, self.embedding_dim))
        return self._encode_texts(nodes)

    def encode_edges(self, edges: List[str]) -> np.ndarray:
        """
        Encode a list of edge attribute strings into vectors.
        Args:
            edges (List[str]): List of edge labels/attributes.

        Returns:
            np.ndarray: Embeddings shape (num_edges, embedding_dim)
        """
        if not edges:
            return np.zeros((0, self.embedding_dim))
        return self._encode_texts(edges)

    def encode_question(self, question: str) -> np.ndarray:
        """
        Encode a question string into a vector.
        Args:
            question (str): User question text.

        Returns:
            np.ndarray: Array of shape (1, embedding_dim)
        """
        q_embed = self._encode_texts([question])
        return q_embed
```

## evaluation.py

```python
## evaluation.py
import numpy as np
import re
from typing import List, Dict, Any, Tuple

class Evaluation:
    def __init__(self, dataset: List[Dict[str, Any]], model_outputs: List[str], ground_truths: List[str], config: Dict):
        """
        Initialize evaluation with dataset, model predictions, ground truths, and config.
        Args:
            dataset (List[Dict]): List of dataset samples, each containing 'graph', etc.
            model_outputs (List[str]): List of predicted answer texts from the model.
            ground_truths (List[str]): List of ground truth answer texts.
            config (Dict): Configuration dictionary with evaluation settings.
        """
        self.dataset = dataset
        self.model_outputs = model_outputs
        self.ground_truths = ground_truths
        # Metrics configuration: e.g., {'accuracy': True, 'hallucination_fraction': True}
        self.metrics_config = config.get('evaluation', {}).get('metrics', {})
        # Batches / steps controls
        self.eval_batch_size = config.get('evaluation', {}).get('eval_batch_size', 16)
        self.eval_steps = config.get('evaluation', {}).get('eval_steps', None)  # Optional
        # Results dictionary to hold computed metrics
        self.results = {}

    def compute_accuracy(self):
        """
        Compute accuracy (or other classification metric) over the dataset.
        Assumes answers are text; compares normalized strings.
        Stores result in self.results.
        """
        correct = 0
        total = len(self.ground_truths)
        for pred, true in zip(self.model_outputs, self.ground_truths):
            if self.match_answers(pred, true):
                correct += 1
        accuracy = correct / total if total > 0 else 0.0
        self.results['accuracy'] = accuracy

    def match_answers(self, pred_answer: str, true_answer: str) -> bool:
        """
        Compare predicted answer with ground truth.
        Uses exact match after lowercasing and stripping.
        Can be extended for more sophisticated metrics.
        """
        return pred_answer.strip().lower() == true_answer.strip().lower()

    def evaluate_hallucination(self):
        """
        Evaluate hallucination metrics based on cited nodes and edges extracted
        from model responses, compared against reference graphs.
        Computes:
           - valid node fraction
           - valid edge fraction
           - full valid graph fraction (both node and edge fully correct)
        """
        valid_node_fractions = []
        valid_edge_fractions = []
        full_valid_flags = []

        for sample_idx, sample in enumerate(self.dataset):
            ref_graph = sample.get('graph', {})
            # Ensure 'nodes' and 'edges' keys exist
            ref_nodes = ref_graph.get('nodes', [])
            ref_edges = ref_graph.get('edges', [])

            pred_response = self.model_outputs[sample_idx]
            cited_nodes, cited_edges = self.extract_citations(pred_response)

            # Validate cited nodes
            ref_node_labels = set()
            # For matching, we use node 'attributes' or 'node_id' (depends on dataset)
            for node in ref_nodes:
                label = node.get('attributes', {}).get('name', '')
                ref_node_labels.add(label.lower())

            valid_nodes_count = sum(1 for n in cited_nodes if n.lower() in ref_node_labels)
            total_cited_nodes = len(cited_nodes) if cited_nodes else 0
            node_fraction = valid_nodes_count / total_cited_nodes if total_cited_nodes > 0 else 0.0
            valid_node_fractions.append(node_fraction)

            # Validate cited edges
            ref_edge_tuples = set()
            for e in ref_edges:
                src = e.get('src')
                dst = e.get('dst')
                attrs = e.get('attributes', {})
                relation = attrs.get('relation', '').lower()
                ref_edge_tuples.add((src, dst, relation))

            valid_edges_count = 0
            total_cited_edges = len(cited_edges) if cited_edges else 0
            for e in cited_edges:
                # Parse edge citation:
                # Assuming citation contains src, dst, relation info
                # For simplicity, assume the citation is formatted as "src->dst:relation" or similar
                # Here, to be robust, parse the text for source, destination, relation
                src, dst, relation = self.parse_edge_citation(e)
                if (src in self.get_node_ids_by_label(ref_nodes, cited_nodes) and
                    dst in self.get_node_ids_by_label(ref_nodes, cited_nodes)):
                    if (src, dst, relation.lower()) in ref_edge_tuples:
                        valid_edges_count += 1
            edge_fraction = valid_edges_count / total_cited_edges if total_cited_edges > 0 else 0.0

            # Record fractions
            self.results.setdefault('valid_node_fraction_list', []).append(node_fraction)
            self.results.setdefault('valid_edge_fraction_list', []).append(edge_fraction)

            # Check if both node and edge citations are fully correct
            if total_cited_nodes == 0:
                nodes_valid_flag = True
            else:
                nodes_valid_flag = (valid_nodes_count == total_cited_nodes)
            if total_cited_edges == 0:
                edges_valid_flag = True
            else:
                edges_valid_flag = (valid_edges_count == total_cited_edges)
            all_valid = nodes_valid_flag and edges_valid_flag
            full_valid_flags.append(1 if all_valid else 0)

        # Aggregate hallucination metrics
        node_frac_mean = np.mean(self.results.get('valid_node_fraction_list', [0]))
        node_frac_std = np.std(self.results.get('valid_node_fraction_list', [0]))
        edge_frac_mean = np.mean(self.results.get('valid_edge_fraction_list', [0]))
        edge_frac_std = np.std(self.results.get('valid_edge_fraction_list', [0]))
        full_valid_mean = np.mean(full_valid_flags)

        self.results['valid_node_fraction'] = node_frac_mean
        self.results['valid_node_fraction_std'] = node_frac_std
        self.results['valid_edge_fraction'] = edge_frac_mean
        self.results['valid_edge_fraction_std'] = edge_frac_std
        self.results['full_graph_fraction'] = full_valid_mean

    def extract_citations(self, response_text: str) -> Tuple[List[str], List[str]]:
        """
        Parse model response to extract cited nodes and edges.
        Assumes the model response mentions node labels or IDs, and edge relations explicitly.
        For example, looks for patterns like:
        - "Nodes: node1, node2, node3"
        - "Edges: edge1, edge2"
        """
        cited_nodes = []
        cited_edges = []

        # Use regex or string parsing based on expected format
        # Example: look for "Nodes" and "Edges" sections
        node_match = re.search(r'Nodes?:\s*(.+)', response_text, re.IGNORECASE)
        if node_match:
            node_str = node_match.group(1)
            # split by comma or semicolon
            node_list = re.split(r'[;,]\s*', node_str)
            cited_nodes = [n.strip() for n in node_list if n.strip()]

        edge_match = re.search(r'Edges?:\s*(.+)', response_text, re.IGNORECASE)
        if edge_match:
            edge_str = edge_match.group(1)
            edge_list = re.split(r'[;,]\s*', edge_str)
            cited_edges = [e.strip() for e in edge_list if e.strip()]

        return cited_nodes, cited_edges

    def parse_edge_citation(self, edge_str: str) -> Tuple[str, str, str]:
        """
        Parse edge citation string to extract src, dst, relation.
        For example, "node1 -> node2:relation" or "node1 -relation-> node2".
        """
        # Simple heuristic implementations:
        # Pattern 1: "node1 -> node2:relation"
        match = re.match(r'(\S+)\s*->\s*(\S+):\s*(\S+)', edge_str)
        if match:
            src, dst, relation = match.groups()
            return src, dst, relation
        # Pattern 2: "node1 -relation-> node2"
        match = re.match(r'(\S+)\s*-\s*(\S+)-\>\s*(\S+)', edge_str)
        if match:
            src, relation, dst = match.groups()
            return src, dst, relation
        # Default fallback: assume "src to dst" with no relation
        parts = re.split(r'\s+to\s+', edge_str)
        if len(parts) == 2:
            return parts[0], parts[1], ''
        # As fallback, return empty
        return '', '', ''

    def get_node_ids_by_label(self, nodes: List[Dict], labels: List[str]) -> set:
        """
        Given nodes and target labels, return set of node ids matching labels.
        """
        node_ids = set()
        label_set = set([lbl.lower() for lbl in labels])
        for node in nodes:
            node_label = node.get('attributes', {}).get('name', '').lower()
            if node_label in label_set:
                node_ids.add(node.get('node_id'))
        return node_ids

    def evaluate_all(self):
        """
        Run all evaluation metrics based on configuration.
        """
        # Compute accuracy if enabled
        if self.metrics_config.get('accuracy', False):
            self.compute_accuracy()
        # Compute hallucination metrics if enabled
        if self.metrics_config.get('hallucination_fraction', False):
            self.evaluate_hallucination()
        return self.results
```

## graph_processor.py

```python
## graph_processor.py
import logging
from typing import Dict, Any

# Set up logging configuration for debug purposes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphProcessor:
    def __init__(self):
        """
        Initialize GraphProcessor.
        Currently, no additional parameters are required.
        """
        pass

    def textualize_graph(self, graph: Dict[str, Any]) -> str:
        """
        Convert a graph dictionary containing nodes and edges into a structured natural language description.
        Args:
            graph (Dict[str, Any]): The input graph with keys like 'nodes' and 'edges'.
                Expected format:
                {
                    "nodes": [{"node_id": int, "attributes": Dict[str, str]}, ...],
                    "edges": [{"src": int, "dst": int, "attributes": Dict[str, str]}, ...]
                }
        Returns:
            str: Textual description of the graph suitable as input for prompt or model input.
        """
        # Validate input graph structure
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        # Build textual node descriptions
        node_descriptions = []
        for node in nodes:
            node_id = node.get("node_id")
            attrs = node.get("attributes", {})
            if node_id is None:
                continue
            # Convert attributes dict to comma-separated string
            attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items() if v)
            if attr_str:
                desc = f"Node {node_id}: {attr_str}"
            else:
                desc = f"Node {node_id}"
            node_descriptions.append(desc)

        # Build textual edge descriptions
        edge_descriptions = []
        for edge in edges:
            src = edge.get("src")
            dst = edge.get("dst")
            attrs = edge.get("attributes", {})
            relation = attrs.get("relation", "")
            # Compose edge description
            if src is not None and dst is not None:
                # Include relation if available
                if relation:
                    desc = f"Node {src} --{relation}--> Node {dst}"
                else:
                    desc = f"{src} -> {dst}"
                edge_descriptions.append(desc)

        # Compose full textual description
        description_parts = []

        if node_descriptions:
            description_parts.append("Nodes:")
            for nd in node_descriptions:
                description_parts.append(f"- {nd}")
        else:
            description_parts.append("No nodes available.")

        if edge_descriptions:
            description_parts.append("Edges:")
            for ed in edge_descriptions:
                description_parts.append(f"- {ed}")
        else:
            description_parts.append("No edges available.")

        # Concatenate all parts into a single string
        textual_description = "\n".join(description_parts)

        # For better prompt integration, optionally return as a paragraph
        # e.g., "This graph contains nodes: ... and edges: ..."
        # Uncomment below if paragraph style is preferred
        # paragraph = "This graph contains " \
        #             f"{len(nodes)} nodes and {len(edges)} edges. " \
        #             "Nodes include: " + ", ".join([f"Node {n.get('node_id')}: {', '.join([f'{k}: {v}' for k, v in n.get('attributes', {}).items()])}" for n in nodes]) + ". " \
        #             "Edges involve: " + ", ".join([f"Node {e.get('src')} --{e.get('attributes', {}).get('relation', '')}--> Node {e.get('dst')}" for e in edges]) + "."

        # Alternatively, for natural language prompt, return a simplified description:
        # e.g.,
        # "The graph contains {num_nodes} nodes and {num_edges} edges. The nodes are: ... The connections are: ..."
        return textual_description
```

## main.py

```python
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
```

## model.py

```python
# model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from typing import Optional, Dict
import logging

class LLMModel:
    def __init__(self, config: Dict):
        """
        Initialize the LLMModel for loading the pretrained Llama2-7B, with optional prompt tuning or LoRA fine-tuning.
        Uses configuration from the provided dictionary, aligning with "config.yaml".
        """
        # Extract configuration options with defaults
        self.model_name: str = config.get("model_name", "Llama2-7B")
        self.model_precision: str = config.get("model_precision", "fp16")
        self.max_input_tokens: int = config.get("max_input_tokens", 512)
        self.prompt_length: int = config.get("prompt_length", 10)  # number of soft prompt tokens
        self.prompt_learning_rate: float = config.get("prompt_learning_rate", 1e-5)
        self.trainable_method: str = config.get("trainable_method", "LoRA")  # "LoRA" or "prompt_tuning"
        # For simplicity, using defaults; optional other configs could be added

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load the base model with FP16 if specified
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.model_precision == "fp16":
            self.model = self.model.half()
        self.model.to(self.device)
        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize prompt tuning or LoRA
        if self.trainable_method == "prompt_tuning":
            # Initialize learnable prompt embedding: shape (prompt_length, hidden_size)
            self.prompt_embeddings = torch.randn(self.prompt_length, self.model.config.hidden_size).to(self.device)
            self.prompt_embeddings = torch.nn.Parameter(self.prompt_embeddings)
            # Optimizer will be external; here, just store
        elif self.trainable_method == "LoRA":
            # Define LoRA config
            lora_cfg = LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
            # Load model with PEFT LoRA extensions
            self.model = get_peft_model(self.model, lora_cfg)
            # Now, only LoRA parameters require grad
            self.model.train()
        else:
            raise ValueError(f"Unsupported trainable_method: {self.trainable_method}")

        # Save parameters for training if needed (training is outside scope here)
        # For inference, prompt embedding tensor is used directly in generate

    def prepare_prompt(self, graph_text: str, question: str) -> Dict:
        """
        Prepare input embeddings by combining optimized prompt tokens (for prompt tuning)
        and the tokenized question + graph description.
        Returns a dict with 'inputs_embeds' and 'attention_mask' for generation.
        """
        # Compose prompt text: e.g., "Graph:\n{graph_text}\nQuestion: {question}\nAnswer:"
        prompt_text = f"Graph:\n{graph_text}\nQuestion: {question}\nAnswer:"
        encoding = self.tokenizer(prompt_text, max_length=self.max_input_tokens,
                                  padding='max_length', truncation=True, return_tensors='pt')

        input_ids = encoding['input_ids'].to(self.device)  # shape: (1, seq_len)
        attention_mask = encoding['attention_mask'].to(self.device)  # shape: (1, seq_len)

        # Get input embeddings
        with torch.no_grad():
            inputs_embeds = self.model.get_input_embeddings()(input_ids).squeeze(0)  # shape: (seq_len, hidden_dim)

        # If prompt tuning, replace first prompt_length tokens with prompt embeddings
        if self.trainable_method == "prompt_tuning":
            # For simplicity, insert prompt embeddings at the start
            # First, ensure the prompt embeddings are of shape (prompt_length, hidden_dim)
            # Replace first prompt_length embeddings
            if self.prompt_embeddings.shape[0] != self.prompt_length:
                # Resize if needed
                self.prompt_embeddings = torch.nn.Parameter(
                    torch.randn(self.prompt_length, self.model.config.hidden_size).to(self.device)
                )
            inputs_embeds[:self.prompt_length, :] = self.prompt_embeddings

        return {'inputs_embeds': inputs_embeds.unsqueeze(0),  # shape: (1, seq_len, hidden_dim)
                'attention_mask': torch.ones(1, inputs_embeds.shape[0], dtype=torch.long).to(self.device)}

    def generate(self, graph_text: str, question: str, max_new_tokens: int = 32) -> str:
        """
        Generate an answer token sequence from the model given textual graph description and question.
        Uses the prepared prompt embeddings as input.
        """
        prompt_inputs = self.prepare_prompt(graph_text, question)
        # Generate tokens
        output_ids = self.model.generate(
            inputs_embeds=prompt_inputs['inputs_embeds'],
            attention_mask=prompt_inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=False  # deterministic; set True if sampling desired
        )
        # Decode output tokens to text
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response

    def save_prompt(self, save_path: str):
        """
        Save prompt embeddings or LoRA weights.
        """
        if self.trainable_method == "prompt_tuning":
            torch.save(self.prompt_embeddings.detach().cpu(), save_path)
        elif self.trainable_method == "LoRA":
            # Save the entire model with PEFT adapter
            self.model.save_pretrained(save_path)
        else:
            raise ValueError(f"Unknown trainable_method: {self.trainable_method}")

    def load_prompt(self, load_path: str):
        """
        Load prompt embeddings or LoRA weights.
        """
        if self.trainable_method == "prompt_tuning":
            loaded = torch.load(load_path).to(self.device)
            self.prompt_embeddings.data.copy_(loaded)
        elif self.trainable_method == "LoRA":
            # Load LoRA adapter weights into the model
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model = get_peft_model(self.model, LoraConfig())
            self.model.load_adapter(load_path)
            self.model.train()
        else:
            raise ValueError(f"Unsupported trainable_method: {self.trainable_method}")
```

## pcst_solver.py

```python
## pcst_solver.py
import networkx as nx
import numpy as np
from typing import Callable

def solve_pcst(
    node_prizes: np.ndarray,
    edge_prizes: np.ndarray,
    edge_cost: float = 1.0
) -> nx.Graph:
    """
    Approximates a Prize-Collecting Steiner Tree (PCST) in a graph to maximize
    total prize of included nodes and edges minus the total edge costs, using
    a near-linear time heuristic approach.

    The function models edge prize > edge cost situations by introducing virtual nodes,
    connecting them with zero-cost edges, and assigning their prizes as (edge_prize - edge_cost).
    It then performs a greedy approximation to solve the PCST problem, ensuring connectivity
    and maximal prize collection.

    Args:
        node_prizes (np.ndarray): Array of shape (N,), prizes for each node.
        edge_prizes (np.ndarray): Array of shape (M,), prizes for each edge.
        edge_cost (float): Uniform cost assigned to each edge in the graph.

    Returns:
        nx.Graph: The connected subgraph satisfying PCST criteria, containing original nodes and edges.
    """
    import networkx as nx

    # Initialize original graph (possibly empty/abstract)
    G = nx.Graph()

    num_nodes = len(node_prizes)
    num_edges = len(edge_prizes)

    # Create graph with original nodes
    # Nodes are labeled from 0 to N-1
    for node_idx in range(num_nodes):
        G.add_node(node_idx, prize=node_prizes[node_idx], virtual=False)

    # For edges, we need to know their structure. As the input is only edge prizes,
    # assume edges are ordered and correspond to some external structure or
    # are constructed beforehand. Here, for the algorithm, we'll process edges
    # as abstract placeholders. In practice, the edges connect specific node pairs.
    # For demonstration, we assume edges are given as a list of (src, dst),
    # or we need to pass the actual edge list; since only prizes are given, we assume
    # an external edge list is available in a real setting.
    # For this implementation, we'll just assume edges as placeholders:
    # e.g., edges are (0,1), (1,2), etc. For realistic use, pass the actual edge list.
    # Here, to comply with "Data structures and interfaces", we will generate dummy edges:
    # For the purpose of the code, assume edges connect nodes with same index or sequentially.
    
    # Placeholder: connect nodes in a chain up to min(num_nodes-1, num_edges-1)
    # In real usage, replace this with actual edge list.
    max_edges = min(num_edges, num_nodes - 1)  # simplistic chain
    for idx in range(max_edges):
        src = idx
        dst = idx+1
        # Assign edge prize
        ep = edge_prizes[idx]
        # Clone the edge into the graph
        G.add_edge(src, dst, prize=ep, virtual=False)
    
    # Now, process edges for virtual nodes
    # For this, iterate over all edges
    virtual_node_id = num_nodes  # start index for virtual nodes
    edges_to_remove = []
    added_virtual_nodes = []

    for u, v, data in list(G.edges(data=True)):
        prize_e = data.get('prize', 0.0)
        if prize_e > edge_cost:
            # Create virtual node to handle this edge
            v_node = virtual_node_id
            virtual_node_id += 1
            # Assign prize to virtual node
            virtual_prize = prize_e - edge_cost
            G.add_node(v_node, prize=virtual_prize, virtual=True)
            added_virtual_nodes.append(v_node)
            # Remove original edge
            edges_to_remove.append((u, v))
            # Add zero-cost edges connecting virtual node to u and v
            G.add_edge(u, v_node, prize=0.0, virtual=True)
            G.add_edge(v, v_node, prize=0.0, virtual=True)

    # Remove edges with prize > edge_cost, replaced by virtual nodes
    for e in edges_to_remove:
        if G.has_edge(*e):
            G.remove_edge(*e)

    # Apply a heuristic for PCST: greedy max prize routing
    # We'll implement a simple approach:
    # Rank nodes and edges by their prize, and greedily include high-prize elements,
    # ensuring connectivity.

    # Extract nodes sorted by prize
    nodes_sorted = sorted(G.nodes(data=True), key=lambda x: x[1].get('prize', 0.0), reverse=True)

    # Initialize subgraph
    subgraph = nx.Graph()
    # Start with highest prize node
    if not nodes_sorted:
        return subgraph  # empty graph if no nodes

    # Keep track of included nodes for connectivity
    included_nodes = set()
    # Initialize with the top node
    top_node_id = nodes_sorted[0][0]
    included_nodes.add(top_node_id)
    subgraph.add_node(top_node_id, **G.nodes[top_node_id])

    # Create a min-heap or sorted list of edges to add
    # Edges connecting included nodes to others
    candidate_edges = []

    # Collect candidate edges
    for u in included_nodes:
        for v, data in G[u].items():
            if v not in included_nodes:
                prize_e = data.get('prize', 0.0)
                edge_prize = prize_e
                candidate_edges.append((edge_prize, u, v, data))

    # Sort candidate edges by their prize
    candidate_edges.sort(key=lambda x: x[0], reverse=True)

    # Greedily include edges/nodes with highest prizes, ensuring connectivity
    while candidate_edges:
        prize_e, u, v, data = candidate_edges.pop(0)
        if v in included_nodes:
            continue
        # Include the node v
        included_nodes.add(v)
        subgraph.add_node(v, **G.nodes[v])
        subgraph.add_edge(u, v, **data)
        # Now, add new candidate edges from v
        for vv, data_v in G[v].items():
            if vv not in included_nodes:
                prize_e_new = data_v.get('prize', 0.0)
                candidate_edges.append((prize_e_new, v, vv, data_v))
        # Re-sort the edges
        candidate_edges.sort(key=lambda x: x[0], reverse=True)

    # Remove virtual nodes and associated edges
    subgraph_nodes = list(subgraph.nodes(data=True))
    for node_id, attr in subgraph_nodes:
        if attr.get('virtual', False):
            subgraph.remove_node(node_id)

    # Ensure connectivity
    if not nx.is_connected(subgraph):
        # Optionally, return the largest connected component
        largest_cc = max(nx.connected_components(subgraph), key=len)
        subgraph = subgraph.subgraph(largest_cc).copy()

    return subgraph
```

## prompt_tuning.py

```python
# prompt_tuning.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Dict, Optional

class PromptTuner:
    def __init__(self, config: Dict):
        """
        Initialize the PromptTuner for prompt tuning or LoRA fine-tuning.

        Args:
            config (Dict): Configuration dictionary containing settings for model, training,
                           prompt length, learning rates, etc.
        """
        # Load configuration parameters with defaults
        self.model_name = config.get("model_name", "Llama2-7B")
        self.model_precision = config.get("model_precision", "fp16")
        self.prompt_length = config.get("prompt_length", 10)
        self.prompt_learning_rate = config.get("prompt_learning_rate", 1e-5)
        self.trainable_method = config.get("trainable_method", "prompt_tuning")  # or "LoRA"
        self.max_input_tokens = config.get("max_input_tokens", 512)

        # Load the base pretrained model with tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
        self.model.to(self.device)
        self.model.eval()

        # Freeze all model weights initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize prompt tokens or LoRA
        if self.trainable_method == "prompt_tuning":
            # Initialize prompt tokens as learnable parameters
            self.prompt_tokens = nn.Parameter(
                torch.randn(self.prompt_length, self.model.config.hidden_size)
            ).to(self.device)
            # Only prompt tokens are trainable
            self.prompt_params = [self.prompt_tokens]
            # Optimizer for prompt tokens
            self.optimizer = optim.AdamW([self.prompt_tokens], lr=self.prompt_learning_rate)
        elif self.trainable_method == "LoRA":
            # Incorporate LoRA modules into the model
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],  # target attention modules
                lora_dropout=0.05,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.train()
            # Only LoRA parameters are trainable
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.prompt_learning_rate
            )
        else:
            raise ValueError(f"Unknown trainable_method: {self.trainable_method}")

    def prepare_prompt(self, textual_graph: str, question: str) -> torch.Tensor:
        """
        Embeds the prompt including the trainable prompt tokens and question, returns input tensor.

        Args:
            textual_graph (str): Textualized graph description.
            question (str): User question text.

        Returns:
            torch.Tensor: Input embeddings tensor ready for model, shape [seq_len, hidden_dim]
        """
        # Compose the prompt text
        prompt_text = f"Graph:\n{textual_graph}\nQuestion: {question}\nAnswer:"
        # Tokenize the prompt text
        encoding = self.tokenizer(prompt_text, max_length=self.max_input_tokens,
                                  padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)  # shape: [1, seq_len]
        attention_mask = encoding['attention_mask'].to(self.device)

        # Convert input_ids to embeddings
        # Get the embedding layer from the model
        with torch.no_grad():
            embedding_layer = self.model.get_input_embeddings()
            input_embeddings = embedding_layer(input_ids).squeeze(0)  # shape: [seq_len, hidden_dim]

        # Replace the initial prompt tokens with learnable prompt embeddings if prompt tuning
        if self.trainable_method == "prompt_tuning":
            # Embed prompt tokens
            prompt_embeddings = self.prompt_tokens  # shape: [prompt_length, hidden_dim]
            # Concatenate prompt embeddings with the rest of input embeddings
            # For simplicity, replace first prompt_length tokens with prompt_embeddings
            # Alternatively, insert prompt embeddings at the beginning
            # Let's replace the first prompt_length tokens
            input_embeddings[:self.prompt_length, :] = prompt_embeddings

        # Return the final input embeddings tensor
        return input_embeddings.unsqueeze(0)  # add batch dim, shape: [1, seq_len, hidden_dim]

    def train_step(self, input_embeddings: torch.Tensor, attention_mask: torch.Tensor, ground_truth_ids: torch.Tensor):
        """
        Perform a single training step: forward, loss, backprop, optimizer step.

        Args:
            input_embeddings (torch.Tensor): Embeddings for input prompt + question, shape: [1, seq_len, hidden_dim]
            attention_mask (torch.Tensor): Attention mask corresponding to input, shape: [1, seq_len]
            ground_truth_ids (torch.Tensor): Token IDs for ground truth answer, shape: [tgt_seq_len]
        """
        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass by feeding input embeddings directly
        # Use model's forward with inputs_embeds instead of input_ids
        # Setup labels for loss comparison
        outputs = self.model(inputs_embeds=input_embeddings,
                             attention_mask=attention_mask,
                             labels=ground_truth_ids)
        loss = outputs.loss

        # Backpropagate only through prompt tokens / LoRA modules
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def generate_response(self, textual_graph: str, question: str, max_new_tokens: int = 32) -> str:
        """
        Generate answer given textual graph and question, incorporating prompt tokens.

        Args:
            textual_graph (str): Textualized graph description.
            question (str): User question.
            max_new_tokens (int): Max tokens to generate in response.

        Returns:
            str: Generated answer text.
        """
        # Prepare prompt with prompt tokens embedded
        input_embeddings = self.prepare_prompt(textual_graph, question)
        # Create attention mask (all ones since embedded tokens)
        seq_len = input_embeddings.shape[1]
        attention_mask = torch.ones((1, seq_len), dtype=torch.long).to(self.device)

        # Generate output ids using model.generate with inputs_embeds
        output_ids = self.model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False  # deterministic
        )

        # Decode generated tokens, skipping prompt tokens if necessary
        response_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response_text

    def save_prompt_parameters(self, save_path: str):
        """
        Save the learned prompt tokens or LoRA weights.

        Args:
            save_path (str): Path to save the parameters.
        """
        if self.trainable_method == "prompt_tuning":
            torch.save(self.prompt_tokens.detach().cpu(), save_path)
        elif self.trainable_method == "LoRA":
            # Save LoRA adapter weights
            self.model.save_pretrained(save_path)
        else:
            raise ValueError("Unknown method for saving parameters.")

    def load_prompt_parameters(self, load_path: str):
        """
        Load prompt tokens or LoRA weights from checkpoint.

        Args:
            load_path (str): Path to load the parameters from.
        """
        if self.trainable_method == "prompt_tuning":
            loaded = torch.load(load_path).to(self.device)
            self.prompt_tokens.data.copy_(loaded)
        elif self.trainable_method == "LoRA":
            # Load LoRA adapter weights
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
            self.model = get_peft_model(self.model, LoraConfig())  # dummy config for loading
            self.model.load_adapter(load_path)
            self.model.train()
        else:
            raise ValueError("Unknown method for loading parameters.")
```

## retrieval.py

```python
## retrieval.py
import numpy as np
import faiss
from typing import Tuple, List
from utils import cosine_similarity

class RetrievalSystem:
    def __init__(self, embedding_dim: int = 1024):
        """
        Initializes the retrieval system with empty FAISS index.
        Args:
            embedding_dim (int): Dimensionality of the node and edge embeddings.
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.embeddings = None  # Will hold all embeddings for reference
        self.id_mapping = None  # Optional: Map index to node/edge IDs or info

    def build_index(self, embeddings: np.ndarray) -> None:
        """
        Builds a FAISS index from the provided embeddings.
        Args:
            embeddings (np.ndarray): The array of all node/edge embeddings, shape (N, d).
        """
        # Normalize embeddings to unit vectors for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        normalized_embeddings = embeddings / norms

        # Create FAISS index for inner product (cosine similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(normalized_embeddings.astype(np.float32))
        self.embeddings = normalized_embeddings
        # Optionally, maintain mapping from index to IDs or source info
        # For now, assume index order corresponds to dataset order

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Search for top-k most similar embeddings to the query vector.
        Args:
            query_vec (np.ndarray): The embedding vector of the query, shape (d,).
            top_k (int): Number of nearest neighbors to retrieve.
        Returns:
            Tuple[List[int], List[float]]: Indices of top-k embeddings and their cosine similarity scores.
        """
        if self.index is None:
            raise ValueError("FAISS index has not been built. Call build_index() first.")

        # Normalize query vector
        norm = np.linalg.norm(query_vec)
        if norm == 0:
            norm = 1
        query_norm = query_vec / norm

        # Search the FAISS index
        distances, indices = self.index.search(query_norm.reshape(1, -1).astype(np.float32), top_k)
        # distances shape: (1, top_k), indices shape: (1, top_k)

        # Flatten outputs
        distances = distances[0]
        indices = indices[0]

        # Convert to Python lists
        return list(indices), list(distances)
```

## train.py

```python
# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from typing import List, Dict, Any
from dataset_loader import DatasetLoader
from embedding import Embedding
from retrieval import RetrievalSystem
from pcst_solver import solve_pcst
from prompt_tuning import PromptTuner
from model import LLMModel
from evaluation import Evaluator

class Trainer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Trainer with configuration, datasets, model, and optimizer.
        Args:
            config (Dict): Configuration dictionary loaded from 'config.yaml'.
        """
        # Set random seed for reproducibility
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load datasets
        dataset_paths = {
            'train': config['dataset']['train_path'],
            'val': config['dataset']['val_path'],
            'test': config['dataset']['test_path']
        }
        self.dataset_loader = DatasetLoader(dataset_paths)
        self.train_data, self.val_data, self.test_data = self.dataset_loader.load_data()

        # Initialize embedding model for nodes/edges/questions
        self.embedding_model = Embedding(
            model_name=config['embedding']['model_name'],
            max_input_tokens=config['embedding']['max_input_tokens'],
            model_precision=config['embedding']['model_precision']
        )

        # Build FAISS index for retrieval
        # Concatenate all node and edge embeddings from dataset for indexing
        all_node_texts = []
        all_edge_texts = []

        # Collect all node/edge attributes for indexing
        for data_point in self.train_data + self.val_data + self.test_data:
            # For indexing, we can use textual descriptions directly or embeddings
            # Here, precompute embeddings for nodes and edges
            for node in data_point['nodes']:
                txt = f"Node: {node['attributes']}"
                all_node_texts.append(txt)
            for edge in data_point['edges']:
                src_str = str(edge['src'])
                dst_str = str(edge['dst'])
                txt = f"Edge: {edge['attributes']} from {src_str} to {dst_str}"
                all_edge_texts.append(txt)
        # Encode all node and edge texts
        node_embeddings = self.embedding_model.encode_nodes(all_node_texts)
        edge_embeddings = self.embedding_model.encode_edges(all_edge_texts)

        # For simplicity, concatenate node and edge embeddings for retrieval ...
        self.retrieval_system = RetrievalSystem(embedding_dim=self.embedding_model.embedding_dim)
        combined_embeddings = np.vstack([node_embeddings, edge_embeddings])
        self.retrieval_system.build_index(combined_embeddings)

        # Initialize model (LLM) with prompt tuning or LoRA
        self.model_cfg = {
            'model_name': config['model']['model_name'],
            'model_precision': config['model']['model_precision'],
            'max_input_tokens': config['model']['max_input_tokens'],
            'prompt_length': config['training']['prompt_length'],
            'prompt_learning_rate': config['training']['prompt_learning_rate']
        }
        self.llm_model = LLMModel(self.model_cfg)

        # Initialize prompt tuner for prompt tuning / LoRA fine-tuning
        self.prompt_tuner = PromptTuner(self.model_cfg)

        # Set optimizer: only prompt tokens or LoRA modules
        if self.prompt_tuner.trainable_method == "prompt_tuning":
            self.optimizer = optim.AdamW([self.prompt_tuner.prompt_tokens], lr=self.model_cfg['prompt_learning_rate'])
        elif self.prompt_tuner.trainable_method == "LoRA":
            # For LoRA, optimizer acts on the LoRA parameters inside the model
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.llm_model.model.parameters()), 
                                           lr=self.model_cfg['prompt_learning_rate'])
        else:
            raise ValueError("Unsupported prompt tuning method specified.")

        # Learning rate scheduler (cosine decay)
        self.num_epochs = config['training']['epochs']
        self.lr = config['training']['learning_rate']
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)

        # Other parameters
        self.batch_size = config['training']['batch_size']
        self.max_input_tokens = config['model']['max_input_tokens']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Instantiate evaluation handler
        self.evaluator = Evaluator()

        # Track best validation metric for checkpointing
        self.best_metric = -float('inf')
        self.checkpoint_path = 'best_model.pt'

    def train(self):
        """
        Run the training loop, including validation and checkpointing.
        """
        for epoch in range(1, self.num_epochs + 1):
            print(f"Epoch {epoch}/{self.num_epochs}")
            self.train_one_epoch()
            val_score = self.validate()
            self.scheduler.step()

            # Save checkpoint if improved
            if val_score > self.best_metric:
                torch.save({
                    'model_state_dict': self.llm_model.model.state_dict(),
                    'prompt_state_dict': getattr(self.prompt_tuner, 'prompt_tokens', None),
                }, self.checkpoint_path)
                self.best_metric = val_score
                print(f"New best model saved with validation score: {val_score}")

    def train_one_epoch(self):
        """
        Train on the training data for one epoch.
        """
        self.llm_model.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle training data
        data = self.train_data
        np.random.shuffle(data)

        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i + self.batch_size]
            # Prepare batch inputs
            input_tensors = []
            target_ids = []
            max_seq_len = 0

            for data_point in batch_data:
                graph_desc = data_point['text_description']
                question = data_point['question']
                answer = data_point['answer']
                # Textualize graph and question
                prompt_text = f"Graph:\n{graph_desc}\nQuestion: {question}\nAnswer:"
                encoding = self.llm_model.tokenizer(prompt_text, max_length=self.max_input_tokens, 
                                                     padding='max_length', truncation=True, return_tensors='pt')
                input_ids = encoding['input_ids']
                attention_mask = encoding['attention_mask']
                # Encode input embeddings with the model's embedding layer
                with torch.no_grad():
                    inputs_embeds = self.llm_model.model.get_input_embeddings()(input_ids).squeeze(0)
                input_tensors.append(inputs_embeds)
                # Prepare target labels (shifted input_ids for teacher forcing)
                labels = input_ids.clone()
                target_ids.append(labels)
                if input_ids.shape[1] > max_seq_len:
                    max_seq_len = input_ids.shape[1]

            # Pad all inputs to max_seq_len
            batch_inputs = []
            batch_mask = []
            batch_labels = []

            for emb, label in zip(input_tensors, target_ids):
                seq_len = emb.shape[0]
                pad_len = max_seq_len - seq_len
                if pad_len > 0:
                    pad_emb = torch.zeros((pad_len, emb.shape[1]), device=emb.device)
                    emb = torch.cat([emb, pad_emb], dim=0)
                    label_pad = torch.full((pad_len,), -100, dtype=torch.long, device=label.device)  # ignore index
                    label = torch.cat([label, label_pad], dim=0)

                batch_inputs.append(emb)
                # Attention mask
                mask = torch.ones(seq_len, device=emb.device)
                if pad_len > 0:
                    mask = torch.cat([mask, torch.zeros(pad_len, device=mask.device)])
                batch_mask.append(mask)
                batch_labels.append(label)

            input_embeds_batch = torch.stack(batch_inputs)  # shape: [batch_size, seq_len, hidden]
            attention_mask_batch = torch.stack(batch_mask)  # shape: [batch_size, seq_len]
            labels_batch = torch.stack(batch_labels)  # shape: [batch_size, seq_len]

            input_embeds_batch = input_embeds_batch.to(self.device)
            attention_mask_batch = attention_mask_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.llm_model.model(
                inputs_embeds=input_embeds_batch,
                attention_mask=attention_mask_batch,
                labels=labels_batch
            )
            loss = outputs.loss
            loss.backward()

            # Optional: clip gradients
            torch.nn.utils.clip_grad_norm_(self.llm_model.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                print(f"Batch {num_batches}: loss {loss.item():.4f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch training loss: {avg_loss:.4f}")

    def validate(self) -> float:
        """
        Run validation, compute metrics, and return validation score.
        """
        self.llm_model.model.eval()
        total_score = 0.0
        total_samples = 0
        hallucination_stats = {'valid_nodes': 0, 'valid_edges': 0, 'full_valid': 0}
        # For simplicity, evaluate on a subset or entire validation data
        for data_point in self.val_data:
            graph_desc = data_point['text_description']
            question = data_point['question']
            answer = data_point['answer']
            # Use G-Retriever to retrieve subgraph, textualize, and generate answer
            generated_text = self._generate_answer(graph_desc, question)
            # Evaluate answer correctness (accuracy/e.g.)
            correct = self.evaluator.compute_accuracy(generated_text, answer)
            total_score += correct
            total_samples += 1

            # Hallucination metrics, if needed, can be evaluated here
            # (Requires explicit extraction from generated_text and ground truth references)

        avg_score = total_score / total_samples if total_samples > 0 else 0
        print(f"Validation accuracy: {avg_score:.4f}")
        return avg_score

    def _generate_answer(self, graph_desc: str, question: str) -> str:
        """
        Uses retrieval, PCST, textualization, and model inference to generate answer.
        """
        # For brevity, placeholder: directly call model.generate with prompt
        return self.llm_model.generate(graph_desc, question)

# Main execution
if __name__ == "__main__":
    import yaml
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # Instantiate and run trainer
    trainer = Trainer(config)
    trainer.train()
```

## utils.py

```python
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
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\G-Retriever\G-Retriever_repo`
