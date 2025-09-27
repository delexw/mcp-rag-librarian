"""
Persistence abstractions following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class PersistenceProvider(ABC):
    """Abstract persistence provider interface (DIP - Dependency Inversion)"""

    @abstractmethod
    def save_embeddings(self, key: str, embeddings: Any, documents: List) -> bool:
        """Save embeddings and documents"""
        pass

    @abstractmethod
    def load_embeddings(self, key: str) -> Optional[Tuple[Any, List, Optional[int]]]:
        """Load embeddings and documents, optionally including dimension"""
        pass

    @abstractmethod
    def save_index(self, key: str, index: Any) -> bool:
        """Save vector index"""
        pass

    @abstractmethod
    def load_index(self, key: str, dimension: int) -> Optional[Any]:
        """Load vector index"""
        pass


class CacheManager:
    """Single Responsibility: Manages cache operations (SRP)"""

    def __init__(self, provider: PersistenceProvider):
        self.provider = provider

    def get_or_create(self, key: str, factory_func, *args, **kwargs):
        """Get from cache or create using factory function"""
        try:
            cached_data = self.provider.load_embeddings(key)
            if cached_data:
                logger.info(f"Loaded from cache: {key}")
                return cached_data
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")

        # Create new data
        data = factory_func(*args, **kwargs)

        # Save to cache
        try:
            self.provider.save_embeddings(key, *data)
            logger.info(f"Saved to cache: {key}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

        return data


class ConfigHasher:
    """Single Responsibility: Handles configuration hashing (SRP)"""

    @staticmethod
    def hash_config(embedding_model: str, chunk_size: int, chunk_overlap: int) -> str:
        """Generate hash for configuration"""
        import hashlib
        config_str = f"{embedding_model}:{chunk_size}:{chunk_overlap}"
        return hashlib.md5(config_str.encode()).hexdigest()


class CacheKeyGenerator:
    """Single Responsibility: Generates cache keys (SRP)"""

    def __init__(self, hasher: ConfigHasher):
        self.hasher = hasher

    def generate_embeddings_key(self, kb_path: str, embedding_model: str,
                               chunk_size: int, chunk_overlap: int) -> str:
        """Generate cache key for embeddings"""
        # Normalize path to ensure consistent cache keys
        from pathlib import Path
        normalized_path = str(Path(kb_path).resolve())
        config_hash = self.hasher.hash_config(embedding_model, chunk_size, chunk_overlap)
        return f"{normalized_path}:embeddings:{config_hash}"

    def generate_index_key(self, kb_path: str, embedding_model: str,
                          chunk_size: int, chunk_overlap: int) -> str:
        """Generate cache key for index"""
        # Normalize path to ensure consistent cache keys
        from pathlib import Path
        normalized_path = str(Path(kb_path).resolve())
        config_hash = self.hasher.hash_config(embedding_model, chunk_size, chunk_overlap)
        return f"{normalized_path}:index:{config_hash}"


class PersistenceStrategy:
    """Open/Closed: Can be extended with new strategies (OCP)"""

    def __init__(self, provider: PersistenceProvider, cache_manager: CacheManager,
                 key_generator: CacheKeyGenerator):
        self.provider = provider
        self.cache_manager = cache_manager
        self.key_generator = key_generator

    async def get_or_create_knowledge_base(self, kb_path: str, embedding_model: str,
                                         chunk_size: int, chunk_overlap: int,
                                         factory_func):
        """Get knowledge base from cache or create new one"""
        logger.info(f"🔄 PERSISTENCE: Starting get_or_create_knowledge_base for {kb_path}")

        embeddings_key = self.key_generator.generate_embeddings_key(
            kb_path, embedding_model, chunk_size, chunk_overlap)
        index_key = self.key_generator.generate_index_key(
            kb_path, embedding_model, chunk_size, chunk_overlap)

        logger.info(f"🔑 PERSISTENCE: Generated keys - embeddings: {embeddings_key}")

        # Try to load from cache
        logger.info("🔍 PERSISTENCE: Attempting to load from cache...")
        try:
            cached_embeddings_data = self.provider.load_embeddings(embeddings_key)
            logger.info(f"📦 PERSISTENCE: Cache load result: {type(cached_embeddings_data)}")
            cached_index = None

            if cached_embeddings_data:
                logger.info("✅ PERSISTENCE: Found cached embeddings, unpacking...")
                embeddings, documents, dimension = cached_embeddings_data
                logger.info(f"📊 PERSISTENCE: Unpacked {len(documents)} docs, dimension: {dimension}")

                # Use cached dimension if available, otherwise fallback
                if dimension:
                    logger.info(f"🎯 PERSISTENCE: Loading index with dimension {dimension}")
                    cached_index = self.provider.load_index(index_key, dimension)
                else:
                    logger.info("🔄 PERSISTENCE: No dimension in cache, trying to get from embeddings")
                    # Fallback: get dimension from embeddings array
                    if hasattr(embeddings, 'shape') and len(embeddings.shape) > 1:
                        dimension = embeddings.shape[1]
                        logger.info(f"📐 PERSISTENCE: Got dimension {dimension} from embeddings shape")
                        cached_index = self.provider.load_index(index_key, dimension)

                if cached_index:
                    logger.info("🎉 PERSISTENCE: Loaded complete knowledge base from cache")
                    return embeddings, documents, cached_index
                else:
                    logger.info("❌ PERSISTENCE: Index not found in cache")
            else:
                logger.info("❌ PERSISTENCE: No cached embeddings found")

        except Exception as e:
            logger.error(f"💥 PERSISTENCE: Cache loading failed: {e}")
            import traceback
            logger.error(f"💥 PERSISTENCE: Traceback: {traceback.format_exc()}")

        # Create new knowledge base
        logger.info("🏗️ PERSISTENCE: Creating new knowledge base via factory...")
        try:
            result = await factory_func()
            if result is None:
                logger.info("🔄 PERSISTENCE: Factory returned None (cache miss skip)")
                return None, [], None
            embeddings, documents, index = result
            logger.info(f"✅ PERSISTENCE: Factory created {len(documents)} docs, index initialized: {index.initialized if hasattr(index, 'initialized') else 'unknown'}")
        except Exception as e:
            logger.error(f"💥 PERSISTENCE: Factory function failed: {e}")
            raise

        # Save to cache
        logger.info("💾 PERSISTENCE: Saving to cache...")
        try:
            logger.info("💾 PERSISTENCE: Saving embeddings...")
            save_embeddings_ok = self.provider.save_embeddings(embeddings_key, embeddings, documents)
            logger.info(f"💾 PERSISTENCE: Save embeddings result: {save_embeddings_ok}")

            logger.info("💾 PERSISTENCE: Saving index...")
            save_index_ok = self.provider.save_index(index_key, index)
            logger.info(f"💾 PERSISTENCE: Save index result: {save_index_ok}")

            logger.info("🎉 PERSISTENCE: Successfully saved knowledge base to cache")
        except Exception as e:
            logger.error(f"💥 PERSISTENCE: Failed to save to cache: {e}")
            import traceback
            logger.error(f"💥 PERSISTENCE: Save traceback: {traceback.format_exc()}")

        logger.info("🏁 PERSISTENCE: Returning knowledge base")
        return embeddings, documents, index