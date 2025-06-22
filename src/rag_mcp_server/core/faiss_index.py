"""
FAISS index module for RAG MCP server.

Handles vector similarity search using FAISS.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS index for efficient similarity search."""
    
    def __init__(self, embedding_dimension: int):
        """
        Initialize FAISS index.
        
        Args:
            embedding_dimension: Dimension of the embedding vectors
        """
        self.dimension = embedding_dimension
        self.index = faiss.IndexFlatIP(embedding_dimension)  # Inner product for cosine similarity
        self.initialized = False
        logger.info(f"Initialized FAISS index with dimension {embedding_dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Numpy array of embeddings (n_docs × embedding_dim)
            
        Raises:
            ValueError: If embeddings array is invalid
            RuntimeError: If adding to index fails
        """
        if embeddings.shape[0] == 0:
            raise ValueError("No embeddings to add to the index")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}")
        
        try:
            # Normalize for cosine similarity
            normalized_embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(normalized_embeddings)
            
            # Add to index
            self.index.add(normalized_embeddings)
            self.initialized = True
            
            logger.info(f"Added {embeddings.shape[0]} vectors to FAISS index")
            
        except Exception as e:
            logger.error(f"Failed to add embeddings to index: {e}")
            raise RuntimeError(f"Failed to add embeddings to index: {e}")
    
    def reset(self) -> None:
        """Reset the index by creating a new empty one."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.initialized = False
        logger.info("FAISS index has been reset")
    
    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
            
        Raises:
            RuntimeError: If index is not initialized or search fails
            ValueError: If query parameters are invalid
        """
        if not self.initialized:
            raise RuntimeError("Index not initialized. Add embeddings first.")
        
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension mismatch: expected {self.dimension}, got {query_embedding.shape[0]}")
        
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        # Ensure top_k doesn't exceed available documents
        max_k = min(top_k, self.index.ntotal)
        
        try:
            # Normalize query for cosine similarity
            query_embedding = query_embedding.astype(np.float32)
            query_embedding = np.expand_dims(query_embedding, axis=0)
            faiss.normalize_L2(query_embedding)
            
            # Search index
            distances, indices = self.index.search(query_embedding, max_k)
            
            return distances[0], indices[0]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search failed: {e}")
    
    def save(self, filepath: Path) -> None:
        """
        Save the index to disk.
        
        Args:
            filepath: Path where to save the index
            
        Raises:
            RuntimeError: If index is not initialized or save fails
        """
        if not self.initialized:
            raise RuntimeError("Cannot save uninitialized index")
        
        try:
            faiss.write_index(self.index, str(filepath))
            logger.info(f"Saved FAISS index to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save index to {filepath}: {e}")
            raise RuntimeError(f"Failed to save index: {e}")
    
    def load(self, filepath: Path) -> None:
        """
        Load an index from disk.
        
        Args:
            filepath: Path to the saved index
            
        Raises:
            FileNotFoundError: If index file doesn't exist
            RuntimeError: If loading fails
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        try:
            self.index = faiss.read_index(str(filepath))
            
            # Verify dimension matches
            if self.index.d != self.dimension:
                raise RuntimeError(f"Loaded index dimension ({self.index.d}) doesn't match expected ({self.dimension})")
            
            self.initialized = True
            logger.info(f"Loaded FAISS index from {filepath} with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to load index from {filepath}: {e}")
            raise RuntimeError(f"Failed to load index: {e}")
    
    def get_stats(self) -> dict:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            "initialized": self.initialized,
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal if self.initialized else 0,
            "index_type": "IndexFlatIP"
        }
