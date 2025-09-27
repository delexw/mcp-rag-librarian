"""
Application State Manager - Single Responsibility for state management.
"""

import logging
from typing import Dict, Any, Optional, Set
from pathlib import Path

from .embedding_service import EmbeddingService
from .document_processor import DocumentProcessor
from .faiss_index import FAISSIndex
from .document_store import DocumentStore
from ..interfaces.knowledge_base_interfaces import (
    ApplicationStateInterface,
    EmbeddingServiceInterface,
    DocumentProcessorInterface,
    SearchIndexInterface,
    DocumentStoreInterface
)

logger = logging.getLogger(__name__)


class ApplicationState(ApplicationStateInterface):
    """
    Manages application state following Single Responsibility Principle.

    Responsibilities:
    - Global state management
    - Component lifecycle
    - State consistency
    """

    def __init__(self):
        """Initialize empty state."""
        self._components: Dict[str, Any] = {
            "embedding_service": None,
            "document_processor": None,
            "faiss_index": None,
            "document_store": None,
            "documents": [],
            "current_kb_path": None,
            "current_cache_key": None,
            "persistence_strategy": None,
        }
        self._initialized_kb_paths: Set[str] = set()

    def update_knowledge_base_components(
        self,
        embedding_service: EmbeddingServiceInterface,
        document_processor: DocumentProcessorInterface,
        faiss_index: SearchIndexInterface,
        document_store: DocumentStoreInterface,
        documents: list,
        kb_path: str,
        cache_key: str
    ):
        """Update all knowledge base components atomically."""
        self._components.update({
            "embedding_service": embedding_service,
            "document_processor": document_processor,
            "faiss_index": faiss_index,
            "document_store": document_store,
            "documents": documents,
            "current_kb_path": kb_path,
            "current_cache_key": cache_key,
        })
        self._initialized_kb_paths.add(cache_key)
        logger.info(f"Updated state for knowledge base: {kb_path}")

    def get_component(self, component_name: str) -> Any:
        """Get a specific component."""
        return self._components.get(component_name)

    def set_component(self, component_name: str, component: Any):
        """Set a specific component."""
        self._components[component_name] = component

    def is_knowledge_base_initialized(self, kb_path: str) -> bool:
        """Check if knowledge base is initialized for given path."""
        return self._components.get("current_kb_path") == kb_path

    def get_current_kb_path(self) -> Optional[str]:
        """Get current knowledge base path."""
        return self._components.get("current_kb_path")

    def get_embedding_service(self) -> Optional[EmbeddingService]:
        """Get current embedding service."""
        return self._components.get("embedding_service")

    def get_document_processor(self) -> Optional[DocumentProcessor]:
        """Get current document processor."""
        return self._components.get("document_processor")

    def get_faiss_index(self) -> Optional[FAISSIndex]:
        """Get current FAISS index."""
        return self._components.get("faiss_index")

    def get_document_store(self) -> Optional[DocumentStore]:
        """Get current document store."""
        return self._components.get("document_store")

    def get_documents(self) -> list:
        """Get current documents."""
        return self._components.get("documents", [])

    def update_search_components(self, faiss_index: FAISSIndex, documents: list):
        """Update only search-related components."""
        self._components.update({
            "faiss_index": faiss_index,
            "documents": documents,
        })
        logger.info("Updated search components")

    def get_persistence_strategy(self):
        """Get persistence strategy."""
        return self._components.get("persistence_strategy")

    def set_persistence_strategy(self, strategy):
        """Set persistence strategy."""
        self._components["persistence_strategy"] = strategy

    def get_stats(self) -> Dict[str, Any]:
        """Get application state statistics."""
        return {
            "initialized_kb_paths": len(self._initialized_kb_paths),
            "current_kb_path": self._components.get("current_kb_path"),
            "has_embedding_service": self._components.get("embedding_service") is not None,
            "has_faiss_index": self._components.get("faiss_index") is not None,
            "document_count": len(self._components.get("documents", [])),
            "has_persistence": self._components.get("persistence_strategy") is not None,
        }