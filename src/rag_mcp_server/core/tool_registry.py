"""
MCP Tool Registry - Open/Closed Principle implementation.

This registry allows adding new tools without modifying existing code,
following the Open/Closed principle.
"""

import logging
from typing import Dict, Any, List, Union
from abc import ABC, abstractmethod

from .response_models import (
    ErrorResponse,
    InitializeKnowledgeBaseResponse,
    KnowledgeBaseStatsResponse,
    ListDocumentsResponse,
    RefreshKnowledgeBaseResponse,
    DocumentInfo,
    ConfigurationInfo,
    SearchResult,
    IndexInfo,
    EmbeddingModelInfo,
    SearchConfigInfo,
    DocumentStoreInfo,
    ChunkingConfigInfo,
    DocumentStats,
    DocumentListItem,
    ToolResponse,
)

logger = logging.getLogger(__name__)

# Common error codes and messages
ERROR_KB_NOT_INITIALIZED = "knowledge_base_not_initialized"
ERROR_INDEX_NOT_INITIALIZED = "index_not_initialized"


def get_kb_not_initialized_message(path: str, use_tool_hint: bool = False) -> str:
    """Get standardized knowledge base not initialized message."""
    hint = " using initialize_knowledge_base tool" if use_tool_hint else ""
    return f"Knowledge base not initialized for path: {path}. Please initialize first{hint}."


def create_error_response(
    operation: str, error_code: str, message: str, path: str = None
) -> ErrorResponse:
    """Create standardized error response for all tools."""
    return ErrorResponse(operation=operation, error=error_code, message=message, path=path)


class MCPTool(ABC):
    """
    Abstract base class for MCP tools (Open/Closed Principle).

    New tools can be created by extending this class without
    modifying existing tool implementations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get tool description."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResponse:
        """Execute the tool with given parameters."""
        pass

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get parameter schema for the tool (optional override)."""
        return {}


class KnowledgeBaseInitializeTool(MCPTool):
    """Tool for initializing knowledge bases."""

    TOOL_NAME = "initialize_knowledge_base"

    def __init__(self, knowledge_base_manager, config_manager, app_state):
        self.kb_manager = knowledge_base_manager
        self.config_manager = config_manager
        self.app_state = app_state

    @property
    def name(self) -> str:
        return self.TOOL_NAME

    @property
    def description(self) -> str:
        return "Initialize a knowledge base from documents."

    async def execute(
        self,
        knowledge_base_path=None,
        embedding_model=None,
        chunk_size=None,
        chunk_overlap=None,
        context=None,
        **kwargs,
    ) -> InitializeKnowledgeBaseResponse:
        """Execute knowledge base initialization."""
        try:
            # Use config manager for parameter resolution
            kb_path_str = self.config_manager.validate_knowledge_base_path(knowledge_base_path)
            embedding_model = self.config_manager.get_embedding_model(embedding_model)
            chunk_size = self.config_manager.get_chunk_size(chunk_size)
            chunk_overlap = self.config_manager.get_chunk_overlap(chunk_overlap)

            # Use knowledge base manager for the actual work
            documents, faiss_index, embedding_service, document_store = (
                await self.kb_manager.initialize_knowledge_base(
                    kb_path_str, embedding_model, chunk_size, chunk_overlap, True, context
                )
            )

            # Update application state
            cache_key = f"{kb_path_str}:{embedding_model}"
            self.app_state.update_knowledge_base_components(
                embedding_service,
                self.kb_manager.get_or_create_document_processor(chunk_size, chunk_overlap),
                faiss_index,
                document_store,
                documents,
                kb_path_str,
                cache_key,
            )

            # Get model info for result
            model_info = embedding_service.get_model_info()
            actual_model_used = model_info.get("actual_model_path", model_info.get("model_name"))
            unique_files = len(set(doc.filename for doc in documents))

            # Create structured response
            response = InitializeKnowledgeBaseResponse(
                path=kb_path_str,
                documents=DocumentInfo(files=unique_files, chunks=len(documents)),
                configuration=ConfigurationInfo(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embedding_model=actual_model_used,
                    embedding_dimension=embedding_service.dimension,
                ),
                message="Knowledge base initialized successfully!",
            )

            if model_info.get("model_mismatch"):
                response.warnings = ["Model mismatch detected"]

            return response

        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            raise


class SemanticSearchTool(MCPTool):
    """Tool for semantic search."""

    TOOL_NAME = "semantic_search"

    def __init__(self, config_manager, app_state):
        self.config_manager = config_manager
        self.app_state = app_state

    @property
    def name(self) -> str:
        return self.TOOL_NAME

    @property
    def description(self) -> str:
        return "Perform semantic search on the knowledge base."

    async def execute(
        self, query: str, knowledge_base_path=None, top_k=None, include_scores=False, **kwargs
    ) -> Union[List[SearchResult], ErrorResponse]:
        """Execute semantic search - returns only results list."""
        try:
            kb_path_str = self.config_manager.validate_knowledge_base_path(knowledge_base_path)
            top_k = self.config_manager.get_top_k(top_k)

            if not self.app_state.is_knowledge_base_initialized(kb_path_str):
                return create_error_response(
                    "semantic_search",
                    ERROR_KB_NOT_INITIALIZED,
                    get_kb_not_initialized_message(kb_path_str),
                    kb_path_str,
                )

            embedding_service = self.app_state.get_embedding_service()
            faiss_index = self.app_state.get_faiss_index()
            documents = self.app_state.get_documents()

            if not faiss_index or not faiss_index.initialized:
                return create_error_response(
                    "semantic_search",
                    ERROR_INDEX_NOT_INITIALIZED,
                    "Knowledge base index is not initialized.",
                    kb_path_str,
                )

            query_embedding = embedding_service.get_embedding(query)
            distances, indices = faiss_index.search(query_embedding, top_k)

            results = []
            for i, (_, idx) in enumerate(zip(distances, indices)):
                if idx < len(documents):
                    doc = documents[idx]
                    content = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content

                    result = SearchResult(
                        rank=i + 1,
                        filename=doc.filename,
                        relative_path=doc.relative_path,
                        content=content,
                    )
                    results.append(result)

            # Return only the results list
            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise


class KnowledgeBaseStatsTool(MCPTool):
    """Tool for getting knowledge base statistics."""

    TOOL_NAME = "get_knowledge_base_stats"

    def __init__(self, config_manager, app_state):
        self.config_manager = config_manager
        self.app_state = app_state

    @property
    def name(self) -> str:
        return self.TOOL_NAME

    @property
    def description(self) -> str:
        return "Get statistics about the knowledge base."

    async def execute(
        self, knowledge_base_path=None, **kwargs
    ) -> Union[KnowledgeBaseStatsResponse, ErrorResponse]:
        """Execute knowledge base stats retrieval."""
        try:
            kb_path_str = self.config_manager.validate_knowledge_base_path(knowledge_base_path)

            if not self.app_state.is_knowledge_base_initialized(kb_path_str):
                return create_error_response(
                    "get_knowledge_base_stats",
                    ERROR_KB_NOT_INITIALIZED,
                    get_kb_not_initialized_message(kb_path_str),
                    kb_path_str,
                )

            # Get components from application state
            embedding_service = self.app_state.get_embedding_service()
            faiss_index = self.app_state.get_faiss_index()
            document_store = self.app_state.get_document_store()
            documents = self.app_state.get_documents()
            document_processor = self.app_state.get_document_processor()

            # Generate statistics
            store_stats = document_store.get_stats()
            index_stats = faiss_index.get_stats()
            model_info = embedding_service.get_model_info()

            unique_files = len(set(doc.filename for doc in documents))
            current_chunk_size = document_processor.chunk_size
            current_chunk_overlap = document_processor.chunk_overlap

            # Create structured response
            return KnowledgeBaseStatsResponse(
                path=kb_path_str,
                documents=DocumentStats(
                    total_files=unique_files,
                    total_chunks=len(documents),
                    average_chunks_per_file=round(
                        len(documents) / unique_files if unique_files > 0 else 0, 1
                    ),
                ),
                chunking_configuration=ChunkingConfigInfo(
                    chunk_size=current_chunk_size, chunk_overlap=current_chunk_overlap
                ),
                index=IndexInfo(
                    status="initialized" if index_stats["initialized"] else "not_initialized",
                    dimension=index_stats["dimension"],
                    total_vectors=index_stats["total_vectors"],
                    index_type=index_stats["index_type"],
                ),
                embedding_model=EmbeddingModelInfo(
                    name=model_info["model_name"],
                    dimension=model_info["dimension"],
                    max_sequence_length=model_info.get("max_sequence_length"),
                ),
                search_configuration=SearchConfigInfo(
                    default_top_k=self.config_manager.get_top_k()
                ),
                document_store=DocumentStoreInfo(
                    database_path=store_stats["database_path"],
                    tracked_documents=store_stats["total_documents"],
                    tracked_chunks=store_stats["total_chunks"],
                    last_modification=store_stats["latest_modification"],
                ),
            )

        except Exception as e:
            logger.error(f"Failed to get knowledge base stats: {e}")
            raise


class ListDocumentsTool(MCPTool):
    """Tool for listing documents in knowledge base."""

    TOOL_NAME = "list_documents"

    def __init__(self, config_manager, app_state):
        self.config_manager = config_manager
        self.app_state = app_state

    @property
    def name(self) -> str:
        return self.TOOL_NAME

    @property
    def description(self) -> str:
        return "List all documents in the knowledge base."

    async def execute(
        self, knowledge_base_path=None, **kwargs
    ) -> Union[ListDocumentsResponse, ErrorResponse]:
        """Execute document listing."""
        try:
            kb_path_str = self.config_manager.validate_knowledge_base_path(knowledge_base_path)

            if not self.app_state.is_knowledge_base_initialized(kb_path_str):
                return create_error_response(
                    "list_documents",
                    ERROR_KB_NOT_INITIALIZED,
                    get_kb_not_initialized_message(kb_path_str),
                    kb_path_str,
                )

            documents = self.app_state.get_documents()
            document_store = self.app_state.get_document_store()

            files_info = {}
            for doc in documents:
                if doc.relative_path not in files_info:
                    files_info[doc.relative_path] = {"chunk_count": 0, "total_content_length": 0}
                files_info[doc.relative_path]["chunk_count"] += 1
                files_info[doc.relative_path]["total_content_length"] += len(doc.content)

            document_list = []
            for relative_path in sorted(files_info.keys()):
                info = files_info[relative_path]
                doc_info = document_store.get_document_info(relative_path)

                doc_item = DocumentListItem(
                    relative_path=relative_path,
                    chunk_count=info["chunk_count"],
                    total_content_length=info["total_content_length"],
                    last_modified=doc_info["last_modified"] if doc_info else None,
                    file_hash=doc_info["file_hash"] if doc_info else None,
                )
                document_list.append(doc_item)

            # Create structured response
            response = ListDocumentsResponse(
                path=kb_path_str, total_documents=len(document_list), documents=document_list
            )

            if not document_list:
                response.success = False
                response.message = "No documents found in the knowledge base"

            return response

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise


class RefreshKnowledgeBaseTool(MCPTool):
    """Tool for refreshing knowledge base."""

    TOOL_NAME = "refresh_knowledge_base"

    def __init__(self, config_manager, app_state, kb_manager):
        self.config_manager = config_manager
        self.app_state = app_state
        self.kb_manager = kb_manager

    @property
    def name(self) -> str:
        return self.TOOL_NAME

    @property
    def description(self) -> str:
        return "Refresh the knowledge base with new or changed documents."

    async def execute(
        self, knowledge_base_path=None, context=None, **kwargs
    ) -> Union[RefreshKnowledgeBaseResponse, ErrorResponse]:
        """Execute knowledge base refresh."""
        try:
            kb_path_str = self.config_manager.validate_knowledge_base_path(knowledge_base_path)

            if not self.app_state.is_knowledge_base_initialized(kb_path_str):
                return create_error_response(
                    "refresh_knowledge_base",
                    ERROR_KB_NOT_INITIALIZED,
                    get_kb_not_initialized_message(kb_path_str, use_tool_hint=True),
                    kb_path_str,
                )

            # Get current configuration from the initialized knowledge base
            embedding_service = self.app_state.get_embedding_service()
            document_processor = self.app_state.get_document_processor()

            model_info = embedding_service.get_model_info()
            embedding_model = model_info.get("model_name", "unknown")
            chunk_size = document_processor.chunk_size
            chunk_overlap = document_processor.chunk_overlap

            if context:
                await context.report_progress(10, 100, "Starting knowledge base refresh...")

            # Use dedicated refresh method that preserves cache and does incremental updates
            documents, faiss_index, embedding_service, document_store, has_changes = (
                await self.kb_manager.refresh_knowledge_base(
                    kb_path_str, embedding_model, chunk_size, chunk_overlap, context
                )
            )

            # Update application state
            cache_key = f"{kb_path_str}:{embedding_model}"
            self.app_state.update_knowledge_base_components(
                embedding_service,
                document_processor,
                faiss_index,
                document_store,
                documents,
                kb_path_str,
                cache_key,
            )

            if context:
                await context.report_progress(100, 100, "Knowledge base refresh completed!")

            unique_files = len(set(doc.filename for doc in documents))
            actual_model_used = model_info.get("actual_model_path", model_info.get("model_name"))
            status_message = (
                "Updated with new or changed documents"
                if has_changes
                else "No changes detected, loaded from cache"
            )

            # Create structured response
            return RefreshKnowledgeBaseResponse(
                path=kb_path_str,
                hasChanges=has_changes,
                documents=DocumentInfo(files=unique_files, chunks=len(documents)),
                configuration=ConfigurationInfo(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embedding_model=actual_model_used,
                    embedding_dimension=embedding_service.dimension,
                ),
                status=status_message,
                message="Knowledge base refreshed successfully!",
            )

        except Exception as e:
            logger.error(f"Failed to refresh knowledge base: {e}")
            raise


class ToolRegistry:
    """
    Registry for MCP tools following Open/Closed Principle.

    New tools can be registered without modifying existing code.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._tools: Dict[str, MCPTool] = {}

    def register_tool(self, tool: MCPTool) -> None:
        """Register a new tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> MCPTool:
        """Get tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        return self._tools[name]

    def get_all_tools(self) -> Dict[str, MCPTool]:
        """Get all registered tools."""
        return self._tools.copy()

    def get_tool_names(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")

    def is_registered(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    async def execute_tool(self, name: str, **kwargs) -> ToolResponse:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        return await tool.execute(**kwargs)
