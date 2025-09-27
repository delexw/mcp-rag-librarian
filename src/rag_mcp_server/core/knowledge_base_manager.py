"""
Knowledge Base Manager - Single Responsibility for KB operations.
"""

import logging
from pathlib import Path
from typing import List, Tuple

from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .faiss_index import FAISSIndex
from .document_store import DocumentStore
from .persistence_factory import PersistenceFactory

logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    """
    Manages knowledge base operations following Single Responsibility Principle.

    Responsibilities:
    - Knowledge base initialization
    - Document processing coordination
    - Index building
    - Cache management
    """

    def __init__(self, embedding_service_factory=None, persistence_strategy=None):
        """Initialize with dependency injection (DIP)."""
        self.embedding_service_factory = embedding_service_factory
        self.persistence_strategy = persistence_strategy or self._create_default_persistence()
        self._embedding_service_cache = {}

    def _create_default_persistence(self):
        """Create default persistence strategy."""
        return PersistenceFactory.create_file_persistence_strategy()

    def get_or_create_embedding_service(self, embedding_model: str) -> EmbeddingService:
        """Get cached embedding service or create new one (SRP)."""
        if embedding_model in self._embedding_service_cache:
            logger.info(f"✅ Found cached EmbeddingService for {embedding_model}")
            return self._embedding_service_cache[embedding_model]

        logger.info(f"🧠 Creating new EmbeddingService for {embedding_model}...")
        if self.embedding_service_factory:
            embedding_service = self.embedding_service_factory(embedding_model)
        else:
            embedding_service = EmbeddingService(embedding_model)

        self._embedding_service_cache[embedding_model] = embedding_service
        logger.info(f"💾 Cached EmbeddingService for {embedding_model}")
        return embedding_service

    async def process_documents_and_build_index(
        self,
        kb_path: Path,
        embedding_service: EmbeddingService,
        document_processor: DocumentProcessor,
        context=None,
    ) -> Tuple[List, FAISSIndex, object]:
        """Process documents and build FAISS index (SRP). Returns embeddings to avoid recomputation."""
        # Load and process documents
        logger.info(f"📚 Loading documents from {kb_path}...")
        documents = document_processor.load_documents(kb_path)
        if not documents:
            raise ValueError(f"No documents found in {kb_path}")
        logger.info(f"✅ Loaded {len(documents)} document chunks")

        if context:
            await context.report_progress(
                40, 100, f"Generating embeddings for {len(documents)} chunks..."
            )

        # Generate embeddings
        logger.info(f"🔢 Generating embeddings for {len(documents)} chunks...")
        texts = [doc.content for doc in documents]
        logger.info(f"📝 Extracted texts, total characters: {sum(len(t) for t in texts)}")

        embeddings = embedding_service.get_embeddings(texts)
        logger.info(f"✅ Generated embeddings with shape: {embeddings.shape}")

        if context:
            await context.report_progress(70, 100, "Building FAISS search index...")

        # Build FAISS index
        logger.info(f"🗂️ Building FAISS index with dimension {embedding_service.dimension}...")
        faiss_index = FAISSIndex(embedding_service.dimension)

        logger.info(f"➕ Adding embeddings to FAISS index...")
        faiss_index.add_embeddings(embeddings)
        logger.info(f"✅ FAISS index built, initialized: {faiss_index.initialized}")

        return documents, faiss_index, embeddings

    def update_document_store(
        self,
        documents: List,
        document_store: DocumentStore,
        document_processor: DocumentProcessor,
        kb_path: Path,
    ):
        """Update document store with current documents (SRP)."""
        logger.info("Updating document store...")
        relative_path_chunks = {}
        for doc in documents:
            relative_path_chunks[doc.relative_path] = (
                relative_path_chunks.get(doc.relative_path, 0) + 1
            )

        # Use document processor's centralized file discovery method
        files = document_processor.find_documents(kb_path)

        for file_path in files:
            relative_path = str(file_path.relative_to(kb_path))
            if relative_path in relative_path_chunks:
                file_hash = document_store.compute_file_hash(file_path)
                chunk_count = relative_path_chunks[relative_path]
                document_store.store_document(file_path, file_hash, chunk_count, relative_path)

    async def load_from_cache_only(
        self, kb_path_str: str, embedding_model: str, chunk_size: int, chunk_overlap: int
    ) -> Tuple[List, FAISSIndex, EmbeddingService, DocumentStore]:
        """Load knowledge base ONLY from cache without any rebuilding (cache-only operation)."""
        kb_path = Path(kb_path_str)

        # Get embedding service first
        embedding_service = self.get_or_create_embedding_service(embedding_model)

        # Generate cache keys
        embeddings_key = self.persistence_strategy.key_generator.generate_embeddings_key(
            kb_path_str, embedding_model, chunk_size, chunk_overlap
        )
        index_key = self.persistence_strategy.key_generator.generate_index_key(
            kb_path_str, embedding_model, chunk_size, chunk_overlap
        )

        # Load directly from cache provider without factory fallback
        cache_result = self.persistence_strategy.provider.load_embeddings(embeddings_key)
        if cache_result is None:
            raise ValueError(f"No cached embeddings found for: {kb_path_str}")

        _, documents, _ = cache_result  # Only need documents from cache
        dimension = embedding_service.dimension

        # Load FAISS index from cache
        faiss_index = self.persistence_strategy.provider.load_index(index_key, dimension)
        if faiss_index is None:
            raise ValueError(f"No cached FAISS index found for: {kb_path_str}")

        # Initialize document store (ensure directory exists)
        kb_path.mkdir(parents=True, exist_ok=True)
        store_path = kb_path / "document_store.db"
        document_store = DocumentStore(str(store_path))

        logger.info(
            f"✅ CACHE-ONLY: Loaded {len(documents)} documents from cache without rebuilding"
        )
        return documents, faiss_index, embedding_service, document_store

    async def update_cache_files(
        self,
        kb_path_str: str,
        embedding_model: str,
        documents: List,
        embeddings_array: object,
        faiss_index: FAISSIndex,
        chunk_size: int,
        chunk_overlap: int,
    ) -> bool:
        """Update cache files with current data (SRP). Uses provided embeddings to avoid recomputation."""
        try:
            logger.info("💾 Saving data to cache...")

            # Use the persistence strategy to save updated embeddings and index
            embeddings_key = self.persistence_strategy.key_generator.generate_embeddings_key(
                kb_path_str, embedding_model, chunk_size, chunk_overlap
            )
            index_key = self.persistence_strategy.key_generator.generate_index_key(
                kb_path_str, embedding_model, chunk_size, chunk_overlap
            )

            # Save embeddings to cache (using provided embeddings instead of recomputing)
            logger.info("💾 Saving embeddings to cache...")
            save_embeddings_ok = self.persistence_strategy.provider.save_embeddings(
                embeddings_key, embeddings_array, documents
            )
            logger.info(f"💾 Save embeddings result: {save_embeddings_ok}")

            # Save index to cache
            logger.info("💾 Saving index to cache...")
            save_index_ok = self.persistence_strategy.provider.save_index(index_key, faiss_index)
            logger.info(f"💾 Save index result: {save_index_ok}")

            if save_embeddings_ok and save_index_ok:
                logger.info("✅ Cache files updated successfully")
                return True
            else:
                logger.warning("⚠️ Some cache files failed to update")
                return False

        except Exception as e:
            logger.warning(f"⚠️ Failed to update cache files: {e}")
            return False

    async def refresh_knowledge_base(
        self,
        kb_path_str: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        context=None,
    ) -> Tuple[List, FAISSIndex, EmbeddingService, DocumentStore]:
        """Refresh knowledge base with incremental updates based on file changes."""
        kb_path = Path(kb_path_str)
        if not kb_path.exists():
            raise ValueError(f"Knowledge base directory does not exist: {kb_path_str}")

        # Check document store for changed files
        store_path = kb_path / "document_store.db"
        document_store = DocumentStore(str(store_path))

        # Get current files and their modification times
        document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        current_files = document_processor.find_documents(kb_path)

        has_changes = False
        for file_path in current_files:
            relative_path = str(file_path.relative_to(kb_path))
            current_hash = document_store.compute_file_hash(file_path)
            stored_info = document_store.get_document_info(relative_path)

            if not stored_info or stored_info["file_hash"] != current_hash:
                has_changes = True
                logger.info(f"📝 REFRESH: Detected changes in {relative_path}")
                break

        if not has_changes:
            logger.info("✅ REFRESH: No changes detected, loading from cache")

        # Get or create embedding service (reuse cached if available)
        embedding_service = self.get_or_create_embedding_service(embedding_model)

        # Use persistence strategy - it will detect changes and rebuild if needed
        embeddings, documents, faiss_index = (
            await self.persistence_strategy.get_or_create_knowledge_base(
                kb_path_str,
                embedding_model,
                chunk_size,
                chunk_overlap,
                lambda: self._create_knowledge_base_factory(
                    kb_path_str, embedding_service, document_processor, context
                ),
            )
        )

        # Update document store with current file info
        self.update_document_store(documents, document_store, document_processor, kb_path)

        # Update cache files if there were changes
        if has_changes:
            await self.update_cache_files(
                kb_path_str,
                embedding_model,
                documents,
                embeddings,
                faiss_index,
                chunk_size,
                chunk_overlap,
            )

        return documents, faiss_index, embedding_service, document_store

    async def initialize_knowledge_base(
        self,
        kb_path_str: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        clean_cache: bool = True,
        context=None,
    ) -> Tuple[List, FAISSIndex, EmbeddingService, DocumentStore]:
        """Initialize knowledge base with all components (orchestration)."""
        kb_path = Path(kb_path_str)
        if not kb_path.exists():
            raise ValueError(f"Knowledge base directory does not exist: {kb_path_str}")

        if clean_cache:
            await self._clean_cache_directory(kb_path, context)

        # Get or create embedding service
        embedding_service = self.get_or_create_embedding_service(embedding_model)

        # Create document processor
        document_processor = DocumentProcessor(chunk_size, chunk_overlap)

        # Use persistence strategy for knowledge base creation
        embeddings, documents, faiss_index = (
            await self.persistence_strategy.get_or_create_knowledge_base(
                kb_path_str,
                embedding_model,
                chunk_size,
                chunk_overlap,
                lambda: self._create_knowledge_base_factory(
                    kb_path_str, embedding_service, document_processor, context
                ),
            )
        )

        # Create document store
        store_path = kb_path / "document_store.db"
        document_store = DocumentStore(str(store_path))

        # Update document store
        self.update_document_store(documents, document_store, document_processor, kb_path)

        # Update cache files
        await self.update_cache_files(
            kb_path_str,
            embedding_model,
            documents,
            embeddings,
            faiss_index,
            chunk_size,
            chunk_overlap,
        )

        return documents, faiss_index, embedding_service, document_store

    def get_or_create_document_processor(
        self, chunk_size: int, chunk_overlap: int
    ) -> DocumentProcessor:
        """Get or create document processor with given settings."""
        return DocumentProcessor(chunk_size, chunk_overlap)

    async def _clean_cache_directory(self, kb_path: Path, context=None):
        """Clean cache directory for fresh start."""
        if context:
            await context.report_progress(5, 100, "Cleaning existing cache files...")

        try:
            import shutil

            cache_dir = kb_path / ".rag_cache"
            if cache_dir.exists():
                logger.info(f"🧹 Cleaning cache directory: {cache_dir}")
                shutil.rmtree(cache_dir)
                logger.info("✅ Cache directory cleaned")
            else:
                logger.info(f"ℹ️ No existing cache directory found at: {cache_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean cache: {e}")

    async def _create_knowledge_base_factory(
        self,
        kb_path: str,
        embedding_service: EmbeddingService,
        document_processor: DocumentProcessor,
        context=None,
    ):
        """Factory function for knowledge base creation."""
        documents, faiss_index, embeddings = await self.process_documents_and_build_index(
            Path(kb_path), embedding_service, document_processor, context
        )

        # Return embeddings computed during index building (no recomputation needed)
        return embeddings, documents, faiss_index
