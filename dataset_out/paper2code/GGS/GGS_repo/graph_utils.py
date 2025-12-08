## graph_utils.py
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

def create_sequence_graph(sequences, knn_neighbors=20, similarity_threshold=None):
    """
    Construct an undirected sequence similarity graph.
    Nodes are sequences, edges connect neighbors based on similarity.
    
    Args:
        sequences (List[str]): List of sequence strings.
        knn_neighbors (int): Number of neighbors for KNN graph.
        similarity_threshold (float or None): Threshold distance to connect nodes.
            If None, build KNN graph.
    
    Returns:
        nx.Graph: Undirected graph with sequences as nodes.
    """
    num_sequences = len(sequences)
    if num_sequences == 0:
        return nx.Graph()

    # Compute pairwise distances
    # Use Levenshtein distance if sequences vary in length, else Hamming for fixed length
    # For efficiency, convert sequences to numpy array of characters if needed
    # Here, default to Levenshtein
    dist_matrix = np.zeros((num_sequences, num_sequences))
    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            dist = levenshtein_distance(sequences[i], sequences[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    G = nx.Graph()
    # Add nodes
    for idx, seq in enumerate(sequences):
        G.add_node(idx, sequence=seq)

    if similarity_threshold is not None:
        # Connect nodes within distance threshold
        for i in range(num_sequences):
            for j in range(i + 1, num_sequences):
                if dist_matrix[i, j] <= similarity_threshold:
                    G.add_edge(i, j, weight=1.0)
    else:
        # Build KNN graph based on distances
        for i in range(num_sequences):
            # Get sorted indices based on distances
            sorted_idx = np.argsort(dist_matrix[i])
            neighbors = 0
            for j in sorted_idx[1:knn_neighbors+1]:
                if i != j:
                    G.add_edge(i, j, weight=1.0)
                    neighbors += 1
    return G

def compute_graph_laplacian(graph, normalized=True):
    """
    Compute the (unnormalized or normalized) graph Laplacian matrix.
    
    Args:
        graph (nx.Graph): Input similarity graph.
        normalized (bool): Whether to compute normalized Laplacian.
    
    Returns:
        scipy.sparse.csr_matrix: Laplacian matrix (sparse).
    """
    # Build adjacency matrix
    A = nx.adjacency_matrix(graph)  # CSR sparse matrix
    degree = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degree)

    if not normalized:
        # Unnormalized Laplacian: L = D - A
        L = D - A
    else:
        # Symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        with np.errstate(divide='ignore'):
            d_inv_sqrt = 1.0 / np.sqrt(degree)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        L = sp.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return L.tocsr()

def perform_label_smoothing(labels, laplacian, gamma):
    """
    Smooth labels over the graph using Tikhonov regularization:
    \hat{Y} = (I + gamma * L)^(-1) * Y
    
    Args:
        labels (np.ndarray): 1D array of labels (fitness scores).
        laplacian (scipy.sparse.csr_matrix): The graph Laplacian matrix.
        gamma (float): Regularization hyperparameter.
    
    Returns:
        np.ndarray: Smoothed labels.
    """
    num_nodes = labels.shape[0]
    I = sp.eye(num_nodes, format='csr')
    M = I + gamma * laplacian
    smoothed_labels = spsolve(M, labels)
    return smoothed_labels

def cluster_sequences(sequences, num_clusters=20):
    """
    Cluster sequences based on Levenshtein distance using hierarchical clustering.
    
    Args:
        sequences (List[str]): List of sequences.
        num_clusters (int): Desired number of clusters.
    
    Returns:
        List[List[str]]: List of clusters, each a list of sequences.
    """
    if len(sequences) == 0:
        return []

    # Efficient clustering with sequence distances
    # Compute condensed distance matrix to feed into clustering
    # Using pairwise_distances with custom metric
    def dist_func(x, y):
        return levenshtein_distance(x, y)

    # Compute full distance matrix
    dist_matrix = pairwise_distances(sequences, metric=dist_func, n_jobs=-1)
    # Convert to condensed form
    # Use AgglomerativeClustering with precomputed distances
    clustering = AgglomerativeClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(dist_matrix)

    clusters_dict = {}
    for seq, lbl in zip(sequences, labels):
        clusters_dict.setdefault(lbl, []).append(seq)
    clusters = list(clusters_dict.values())
    return clusters

def select_top_per_cluster(sequences, fitnesses, clusters):
    """
    Select the sequence with highest fitness from each cluster.
    
    Args:
        sequences (List[str]): List of sequences.
        fitnesses (np.ndarray): Corresponding predicted fitness scores.
        clusters (List[List[str]]): Clusters of sequences.
    
    Returns:
        List[str]: Top sequences per cluster.
    """
    top_sequences = []
    seq_to_fit = {seq: fit for seq, fit in zip(sequences, fitnesses)}
    for cluster in clusters:
        # Find sequence with max fitness in cluster
        max_seq = max(cluster, key=lambda s: seq_to_fit[s])
        top_sequences.append(max_seq)
    return top_sequences
