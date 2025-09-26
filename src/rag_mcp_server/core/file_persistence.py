"""
File-based persistence implementation.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Any
import logging

from .persistence import PersistenceProvider
from .faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


class FilePersistenceProvider(PersistenceProvider):
    """File-based persistence implementation (LSP - Liskov Substitution)"""

    def __init__(self, cache_dir_name: str = ".rag_cache"):
        self.cache_dir_name = cache_dir_name

    def _get_cache_dir(self, kb_path: str) -> Path:
        """Get cache directory for a knowledge base"""
        cache_dir = Path(kb_path) / self.cache_dir_name
        cache_dir.mkdir(exist_ok=True)
        return cache_dir

    def _get_embeddings_file(self, kb_path: str, key: str) -> Path:
        """Get embeddings cache file path"""
        cache_dir = self._get_cache_dir(kb_path)
        # Extract config hash from key
        config_hash = key.split(':')[-1]
        return cache_dir / f"embeddings_{config_hash}.pkl"

    def _get_index_file(self, kb_path: str, key: str) -> Path:
        """Get index cache file path"""
        cache_dir = self._get_cache_dir(kb_path)
        # Extract config hash from key
        config_hash = key.split(':')[-1]
        return cache_dir / f"faiss_index_{config_hash}.bin"

    def save_embeddings(self, key: str, embeddings: Any, documents: List) -> bool:
        """Save embeddings and documents to file"""
        try:
            # Extract kb_path from key
            kb_path = key.split(':')[0]
            cache_file = self._get_embeddings_file(kb_path, key)

            # Store dimension with embeddings to avoid recreating EmbeddingService
            dimension = embeddings.shape[1] if hasattr(embeddings, 'shape') else None

            cache_data = {
                "embeddings": embeddings,
                "documents": documents,
                "key": key,
                "dimension": dimension  # Store dimension to avoid recreating embedding service
            }

            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            logger.info(f"Saved embeddings to {cache_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            return False

    def load_embeddings(self, key: str) -> Optional[Tuple[Any, List, Optional[int]]]:
        """Load embeddings and documents from file, including dimension"""
        try:
            # Extract kb_path from key
            kb_path = key.split(':')[0]
            cache_file = self._get_embeddings_file(kb_path, key)

            if not cache_file.exists():
                return None

            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Validate key matches
            if cache_data.get("key") != key:
                logger.warning(f"Cache key mismatch for {cache_file}")
                return None

            logger.info(f"Loaded embeddings from {cache_file}")
            dimension = cache_data.get("dimension")
            return cache_data["embeddings"], cache_data["documents"], dimension

        except Exception as e:
            logger.warning(f"Failed to load embeddings: {e}")
            return None

    def save_index(self, key: str, index: FAISSIndex) -> bool:
        """Save FAISS index to file"""
        try:
            # Extract kb_path from key
            kb_path = key.split(':')[0]
            index_file = self._get_index_file(kb_path, key)

            index.save(index_file)
            logger.info(f"Saved FAISS index to {index_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False

    def load_index(self, key: str, dimension: int) -> Optional[FAISSIndex]:
        """Load FAISS index from file"""
        try:
            # Extract kb_path from key
            kb_path = key.split(':')[0]
            index_file = self._get_index_file(kb_path, key)

            if not index_file.exists():
                return None

            faiss_index = FAISSIndex(dimension)
            faiss_index.load(index_file)

            logger.info(f"Loaded FAISS index from {index_file}")
            return faiss_index

        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return None


class S3PersistenceProvider(PersistenceProvider):
    """S3-based persistence implementation (OCP - Open/Closed Extension)"""

    def __init__(self, bucket_name: str, prefix: str = "rag_cache"):
        self.bucket_name = bucket_name
        self.prefix = prefix
        # Would implement boto3 client here

    def save_embeddings(self, key: str, embeddings: Any, documents: List) -> bool:
        """Save embeddings to S3 (synchronous placeholder)"""
        raise NotImplementedError("S3 persistence not implemented")

    def load_embeddings(self, key: str) -> Optional[Tuple[Any, List, Optional[int]]]:
        """Load embeddings from S3 (synchronous placeholder)"""
        raise NotImplementedError("S3 persistence not implemented")

    def save_index(self, key: str, index: Any) -> bool:
        """Save FAISS index to S3 (synchronous placeholder)"""
        raise NotImplementedError("S3 persistence not implemented")

    def load_index(self, key: str, dimension: int) -> Optional[Any]:
        """Load FAISS index from S3 (synchronous placeholder)"""
        raise NotImplementedError("S3 persistence not implemented")