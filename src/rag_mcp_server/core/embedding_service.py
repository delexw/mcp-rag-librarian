"""
Embedding service module for RAG MCP server.

Handles text embedding operations using SentenceTransformers.
"""

import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles document embedding operations using SentenceTransformers."""
    
    def __init__(self, model_name: str):
        """
        Initialize the embedding service with the specified model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            
        Raises:
            RuntimeError: If the model fails to initialize
        """
        try:
            self.model_name = model_name
            logger.info(f"Attempting to load embedding model: {model_name}")
            
            # Load the model with explicit logging
            self.model = SentenceTransformer(model_name)
            
            # Verify the model loaded correctly by checking its actual name
            actual_model_name = getattr(self.model, '_model_name', None) or getattr(self.model, 'model_name', None)
            if actual_model_name and actual_model_name != model_name:
                logger.warning(f"Requested model '{model_name}' but loaded '{actual_model_name}'")
            
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            # Additional verification - get the actual transformer model info
            try:
                if hasattr(self.model, '_modules') and hasattr(self.model._modules, 'transformer'):
                    transformer_name = self.model._modules['transformer'].auto_model.config.name_or_path
                    logger.info(f"Transformer model path: {transformer_name}")
                    if transformer_name != model_name:
                        logger.warning(f"Transformer model path '{transformer_name}' differs from requested '{model_name}'")
            except Exception as e:
                logger.debug(f"Could not verify transformer model path: {e}")
            
            logger.info(f"Successfully initialized embedding model: {model_name} (dimension: {self.dimension})")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model '{model_name}': {e}")
            logger.info("Attempting fallback to all-MiniLM-L6-v2...")
            
            try:
                # Fallback to a known working model
                fallback_model = "all-MiniLM-L6-v2"
                self.model_name = fallback_model
                self.model = SentenceTransformer(fallback_model)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.warning(f"Fallback successful: Using {fallback_model} instead of {model_name}")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                raise RuntimeError(f"Failed to initialize any embedding model. Original error: {e}, Fallback error: {fallback_error}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding for the given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            ValueError: If text is empty
            RuntimeError: If embedding computation fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
            
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            return embedding
        except Exception as e:
            logger.error(f"Failed to compute embedding for text: {e}")
            raise RuntimeError(f"Embedding computation failed: {e}")
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Compute embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Array of embedding vectors (n_texts × embedding_dim)
            
        Raises:
            ValueError: If texts list is empty
            RuntimeError: If embedding computation fails
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Filter out empty texts and keep track of indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            raise ValueError("No valid texts found after filtering empty strings")
        
        try:
            embeddings = self.model.encode(
                valid_texts, 
                batch_size=batch_size, 
                show_progress_bar=len(valid_texts) > 100  # Only show progress for large batches
            )
            
            # Create full array with zeros for empty texts
            full_embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for i, valid_idx in enumerate(valid_indices):
                full_embeddings[valid_idx] = embeddings[i]
            
            logger.info(f"Generated embeddings for {len(valid_texts)}/{len(texts)} texts using model: {self.model_name}")
            return full_embeddings
            
        except Exception as e:
            logger.error(f"Failed to compute embeddings for {len(texts)} texts: {e}")
            raise RuntimeError(f"Batch embedding computation failed: {e}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the current embedding model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_sequence_length": getattr(self.model, 'max_seq_length', None)
        }
        
        # Try to get the actual underlying model name for verification
        try:
            if hasattr(self.model, '_modules') and hasattr(self.model._modules, 'transformer'):
                actual_model_path = self.model._modules['transformer'].auto_model.config.name_or_path
                info["actual_model_path"] = actual_model_path
                if actual_model_path != self.model_name:
                    info["model_mismatch"] = True
                    logger.warning(f"Model name mismatch: requested '{self.model_name}' but using '{actual_model_path}'")
        except Exception as e:
            logger.debug(f"Could not retrieve actual model path: {e}")
        
        return info
