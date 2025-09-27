"""
Knowledge Base Interfaces - Interface Segregation Principle.

These interfaces define focused contracts for different aspects
of knowledge base operations.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any
from pathlib import Path


class EmbeddingServiceInterface(ABC):
    """Interface for embedding services (ISP)."""

    @abstractmethod
    def get_embedding(self, text: str) -> Any:
        """Get embedding for a single text."""
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> Any:
        """Get embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Get model information."""
        pass


class DocumentProcessorInterface(ABC):
    """Interface for document processing (ISP)."""

    @abstractmethod
    def load_documents(self, path: Path) -> List[Any]:
        """Load documents from path."""
        pass

    @abstractmethod
    def find_documents(self, path: Path) -> List[Path]:
        """Find document files in path."""
        pass

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Get chunk size."""
        pass

    @property
    @abstractmethod
    def chunk_overlap(self) -> int:
        """Get chunk overlap."""
        pass


class SearchIndexInterface(ABC):
    """Interface for search indexes (ISP)."""

    @abstractmethod
    def add_embeddings(self, embeddings: Any) -> None:
        """Add embeddings to index."""
        pass

    @abstractmethod
    def search(self, query_embedding: Any, top_k: int) -> Tuple[Any, Any]:
        """Search for similar embeddings."""
        pass

    @property
    @abstractmethod
    def initialized(self) -> bool:
        """Check if index is initialized."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get index dimension."""
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Get index statistics."""
        pass


class DocumentStoreInterface(ABC):
    """Interface for document storage (ISP)."""

    @abstractmethod
    def store_document(self, file_path: Path, file_hash: str, chunk_count: int, relative_path: str) -> None:
        """Store document metadata."""
        pass

    @abstractmethod
    def is_document_changed(self, file_path: Path, relative_path: str) -> Tuple[bool, str]:
        """Check if document has changed."""
        pass

    @abstractmethod
    def remove_document(self, relative_path: str) -> None:
        """Remove document from store."""
        pass

    @abstractmethod
    def get_all_document_names(self) -> set:
        """Get all stored document names."""
        pass

    @abstractmethod
    def get_document_info(self, relative_path: str) -> Optional[dict]:
        """Get document information."""
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Get store statistics."""
        pass


class CacheManagerInterface(ABC):
    """Interface for cache management (ISP)."""

    @abstractmethod
    async def save_embeddings(self, key: str, embeddings: Any, documents: List[Any]) -> bool:
        """Save embeddings to cache."""
        pass

    @abstractmethod
    async def load_embeddings(self, key: str) -> Tuple[Any, List[Any]]:
        """Load embeddings from cache."""
        pass

    @abstractmethod
    async def save_index(self, key: str, index: SearchIndexInterface) -> bool:
        """Save search index to cache."""
        pass

    @abstractmethod
    async def load_index(self, key: str) -> SearchIndexInterface:
        """Load search index from cache."""
        pass


class KnowledgeBaseManagerInterface(ABC):
    """Interface for knowledge base management (ISP)."""

    @abstractmethod
    async def initialize_knowledge_base(
        self,
        kb_path: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        clean_cache: bool = True,
        context=None
    ) -> Tuple[List[Any], SearchIndexInterface, EmbeddingServiceInterface, DocumentStoreInterface]:
        """Initialize knowledge base."""
        pass

    @abstractmethod
    async def refresh_knowledge_base(
        self,
        kb_path: str,
        context=None
    ) -> str:
        """Refresh knowledge base with changes."""
        pass

    @abstractmethod
    async def search_knowledge_base(
        self,
        query: str,
        kb_path: str,
        top_k: int,
        include_scores: bool = False
    ) -> str:
        """Search knowledge base."""
        pass


class ConfigManagerInterface(ABC):
    """Interface for configuration management (ISP)."""

    @abstractmethod
    def get_value(self, key: str, provided_value: Optional[Any] = None) -> Any:
        """Get configuration value with precedence."""
        pass

    @abstractmethod
    def validate_knowledge_base_path(self, provided_path: Optional[str] = None) -> str:
        """Validate and resolve knowledge base path."""
        pass

    @abstractmethod
    def set_override(self, key: str, value: Any) -> None:
        """Set configuration override."""
        pass


class ApplicationStateInterface(ABC):
    """Interface for application state management (ISP)."""

    @abstractmethod
    def update_knowledge_base_components(
        self,
        embedding_service: EmbeddingServiceInterface,
        document_processor: DocumentProcessorInterface,
        faiss_index: SearchIndexInterface,
        document_store: DocumentStoreInterface,
        documents: list,
        kb_path: str,
        cache_key: str
    ) -> None:
        """Update knowledge base components."""
        pass

    @abstractmethod
    def is_knowledge_base_initialized(self, kb_path: str) -> bool:
        """Check if knowledge base is initialized."""
        pass

    @abstractmethod
    def get_component(self, component_name: str) -> Any:
        """Get application component."""
        pass