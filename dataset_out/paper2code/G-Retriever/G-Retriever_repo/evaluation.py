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
