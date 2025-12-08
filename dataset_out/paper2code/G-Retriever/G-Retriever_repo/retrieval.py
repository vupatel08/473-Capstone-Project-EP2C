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
