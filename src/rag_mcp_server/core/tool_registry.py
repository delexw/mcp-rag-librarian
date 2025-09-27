"""
MCP Tool Registry - Open/Closed Principle implementation.

This registry allows adding new tools without modifying existing code,
following the Open/Closed principle.
"""

import logging
from typing import Dict, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


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
    async def execute(self, **kwargs) -> str:
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

    async def execute(self, knowledge_base_path=None, embedding_model=None,
                     chunk_size=None, chunk_overlap=None, context=None, **kwargs) -> str:
        """Execute knowledge base initialization."""
        try:
            # Use config manager for parameter resolution
            kb_path_str = self.config_manager.validate_knowledge_base_path(knowledge_base_path)
            embedding_model = self.config_manager.get_embedding_model(embedding_model)
            chunk_size = self.config_manager.get_chunk_size(chunk_size)
            chunk_overlap = self.config_manager.get_chunk_overlap(chunk_overlap)

            # Use knowledge base manager for the actual work
            documents, faiss_index, embedding_service, document_store = await self.kb_manager.initialize_knowledge_base(
                kb_path_str, embedding_model, chunk_size, chunk_overlap, True, context
            )

            # Update application state
            cache_key = f"{kb_path_str}:{embedding_model}"
            self.app_state.update_knowledge_base_components(
                embedding_service, self.kb_manager.get_or_create_document_processor(chunk_size, chunk_overlap),
                faiss_index, document_store, documents, kb_path_str, cache_key
            )

            # Get model info for result
            model_info = embedding_service.get_model_info()
            actual_model_used = model_info.get("actual_model_path", model_info.get("model_name"))
            unique_files = len(set(doc.filename for doc in documents))

            result_message = f"""Knowledge base initialized successfully!
- Path: {kb_path_str}
- Documents: {unique_files} files
- Chunks: {len(documents)} (chunk_size: {chunk_size}, overlap: {chunk_overlap})
- Embedding model: {actual_model_used}
- Embedding dimension: {embedding_service.dimension}"""

            if model_info.get("model_mismatch"):
                result_message += f"\n⚠️  WARNING: Model mismatch detected"

            return result_message

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

    async def execute(self, query: str, knowledge_base_path=None,
                     top_k=None, include_scores=False, **kwargs) -> str:
        """Execute semantic search."""
        try:
            kb_path_str = self.config_manager.validate_knowledge_base_path(knowledge_base_path)
            top_k = self.config_manager.get_top_k(top_k)

            if not self.app_state.is_knowledge_base_initialized(kb_path_str):
                return f"Knowledge base not initialized for path: {kb_path_str}. Please initialize first."

            embedding_service = self.app_state.get_embedding_service()
            faiss_index = self.app_state.get_faiss_index()
            documents = self.app_state.get_documents()

            if not faiss_index or not faiss_index.initialized:
                return "Knowledge base index is not initialized."

            query_embedding = embedding_service.get_embedding(query)
            distances, indices = faiss_index.search(query_embedding, top_k)

            results = []
            for i, (distance, idx) in enumerate(zip(distances, indices)):
                if idx < len(documents):
                    doc = documents[idx]
                    result = {
                        "rank": i + 1,
                        "filename": doc.filename,
                        "relative_path": doc.relative_path,
                        "chunk_index": doc.chunk_index,
                        "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    }
                    if include_scores:
                        result["similarity_score"] = float(distance)

                    results.append(result)

            if not results:
                return "No relevant documents found for the query."

            response_text = f"Found {len(results)} relevant document chunks for query: '{query}' (top_k={top_k})\n\n"

            for result in results:
                response_text += f"**Rank {result['rank']}** - {result['relative_path']} (chunk {result['chunk_index']})\n"
                if include_scores:
                    response_text += f"Similarity Score: {result['similarity_score']:.4f}\n"
                response_text += f"Content: {result['content']}\n\n"

            return response_text

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

    async def execute(self, knowledge_base_path=None, **kwargs) -> str:
        """Execute knowledge base stats retrieval."""
        try:
            kb_path_str = self.config_manager.validate_knowledge_base_path(knowledge_base_path)

            if not self.app_state.is_knowledge_base_initialized(kb_path_str):
                return f"Knowledge base not initialized for path: {kb_path_str}. Please initialize first."

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

            stats_text = f"""Knowledge Base Statistics:

**Path:** {kb_path_str}

**Documents:**
- Total files: {unique_files}
- Total chunks: {len(documents)}
- Average chunks per file: {len(documents) / unique_files if unique_files > 0 else 0:.1f}

**Chunking Configuration:**
- Chunk size: {current_chunk_size} characters
- Chunk overlap: {current_chunk_overlap} characters

**Index:**
- Status: {'Initialized' if index_stats['initialized'] else 'Not initialized'}
- Dimension: {index_stats['dimension']}
- Vectors: {index_stats['total_vectors']}
- Type: {index_stats['index_type']}

**Embedding Model:**
- Model: {model_info['model_name']}
- Dimension: {model_info['dimension']}
- Max sequence length: {model_info.get('max_sequence_length', 'N/A')}

**Search Configuration:**
- Default top_k: {self.config_manager.get_top_k()}

**Document Store:**
- Database: {store_stats['database_path']}
- Tracked documents: {store_stats['total_documents']}
- Tracked chunks: {store_stats['total_chunks']}
- Last modification: {store_stats['latest_modification'] or 'N/A'}
"""

            return stats_text

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

    async def execute(self, knowledge_base_path=None, **kwargs) -> str:
        """Execute document listing."""
        try:
            kb_path_str = self.config_manager.validate_knowledge_base_path(knowledge_base_path)

            if not self.app_state.is_knowledge_base_initialized(kb_path_str):
                return f"Knowledge base not initialized for path: {kb_path_str}. Please initialize first."

            documents = self.app_state.get_documents()
            document_store = self.app_state.get_document_store()

            files_info = {}
            for doc in documents:
                if doc.relative_path not in files_info:
                    files_info[doc.relative_path] = {"chunk_count": 0, "total_content_length": 0}
                files_info[doc.relative_path]["chunk_count"] += 1
                files_info[doc.relative_path]["total_content_length"] += len(doc.content)

            if not files_info:
                return "No documents found in the knowledge base."

            response_text = f"Documents in knowledge base: {kb_path_str}\n\n"

            for relative_path in sorted(files_info.keys()):
                info = files_info[relative_path]
                doc_info = document_store.get_document_info(relative_path)

                response_text += f"**{relative_path}**\n"
                response_text += f"  - Chunks: {info['chunk_count']}\n"
                response_text += f"  - Total content length: {info['total_content_length']:,} characters\n"

                if doc_info:
                    response_text += f"  - Last modified: {doc_info['last_modified']}\n"
                    response_text += f"  - File hash: {doc_info['file_hash'][:16]}...\n"

                response_text += "\n"

            return response_text

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

    async def execute(self, knowledge_base_path=None, context=None, **kwargs) -> str:
        """Execute knowledge base refresh."""
        try:
            kb_path_str = self.config_manager.validate_knowledge_base_path(knowledge_base_path)

            if not self.app_state.is_knowledge_base_initialized(kb_path_str):
                return f"Knowledge base not initialized for path: {kb_path_str}. Please initialize first using initialize_knowledge_base tool."

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
            documents, faiss_index, embedding_service, document_store = await self.kb_manager.refresh_knowledge_base(
                kb_path_str, embedding_model, chunk_size, chunk_overlap, context
            )

            # Update application state
            cache_key = f"{kb_path_str}:{embedding_model}"
            self.app_state.update_knowledge_base_components(
                embedding_service, document_processor, faiss_index, document_store, documents, kb_path_str, cache_key
            )

            if context:
                await context.report_progress(100, 100, "Knowledge base refresh completed!")

            unique_files = len(set(doc.filename for doc in documents))
            actual_model_used = model_info.get("actual_model_path", model_info.get("model_name"))

            result_message = f"""Knowledge base refreshed successfully!
- Path: {kb_path_str}
- Documents: {unique_files} files
- Chunks: {len(documents)} (chunk_size: {chunk_size}, overlap: {chunk_overlap})
- Embedding model: {actual_model_used}
- Embedding dimension: {embedding_service.dimension}
- Status: Updated with any new or changed documents"""

            return result_message

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

    async def execute_tool(self, name: str, **kwargs) -> str:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        return await tool.execute(**kwargs)