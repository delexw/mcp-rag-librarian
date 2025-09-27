"""
Simplified MCP server for RAG operations using SOLID architecture.

This simplified version delegates to SOLID components for better maintainability.
"""

import logging
import argparse
from pathlib import Path
from typing import Union

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Context

# SOLID architecture components
from .core.service_container import service_container
from .core.config_manager import ConfigManager
from .core.application_state import ApplicationState
from .core.knowledge_base_manager import KnowledgeBaseManager
from .core.tool_registry import (
    ToolRegistry,
    KnowledgeBaseInitializeTool,
    SemanticSearchTool,
    KnowledgeBaseStatsTool,
    ListDocumentsTool,
    RefreshKnowledgeBaseTool
)
from .core.persistence_factory import PersistenceFactory

# Interfaces
from .interfaces.knowledge_base_interfaces import (
    ConfigManagerInterface,
    ApplicationStateInterface,
    KnowledgeBaseManagerInterface
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server instance
app = FastMCP("rag-server")

# HARDCODED DEFAULTS - NO CONFIG FILE
DEFAULT_VALUES = {
    "embedding_model": "ibm-granite/granite-embedding-278m-multilingual",
    "chunk_size": 500,
    "chunk_overlap": 200,
    "top_k": 7,
    "max_batch_size": 32,
}

logger.info(f"🚀 RAG MCP SERVER - Using SOLID architecture with defaults: {DEFAULT_VALUES}")


class RAGServer:
    """
    RAG server using SOLID architecture.

    This delegates all complex logic to specialized SOLID components.
    """

    def __init__(self):
        """Initialize server with SOLID components."""
        self._setup_solid_dependencies()

    def _setup_solid_dependencies(self):
        """Setup SOLID architecture dependencies."""
        # Configuration management
        config_manager = ConfigManager(DEFAULT_VALUES)
        service_container.register_instance(ConfigManagerInterface, config_manager)

        # Application state
        app_state = ApplicationState()
        service_container.register_instance(ApplicationStateInterface, app_state)

        # Knowledge base management
        kb_manager = KnowledgeBaseManager(
            persistence_strategy=PersistenceFactory.create_file_persistence_strategy()
        )
        service_container.register_instance(KnowledgeBaseManagerInterface, kb_manager)

        # Tool registry
        tool_registry = ToolRegistry()
        tool_registry.register_tool(KnowledgeBaseInitializeTool(kb_manager, config_manager, app_state))
        tool_registry.register_tool(SemanticSearchTool(config_manager, app_state))
        tool_registry.register_tool(KnowledgeBaseStatsTool(config_manager, app_state))
        tool_registry.register_tool(ListDocumentsTool(config_manager, app_state))
        tool_registry.register_tool(RefreshKnowledgeBaseTool(config_manager, app_state, kb_manager))
        service_container.register_instance(ToolRegistry, tool_registry)

    def get_config_manager(self) -> ConfigManagerInterface:
        """Get configuration manager."""
        return service_container.resolve(ConfigManagerInterface)

    def get_app_state(self) -> ApplicationStateInterface:
        """Get application state."""
        return service_container.resolve(ApplicationStateInterface)

    def get_tool_registry(self) -> ToolRegistry:
        """Get tool registry."""
        return service_container.resolve(ToolRegistry)

    def configure_from_args(self, args):
        """Configure server from command line arguments."""
        config_manager = self.get_config_manager()

        if args.knowledge_base:
            kb_path = Path(args.knowledge_base)
            if not kb_path.exists():
                raise ValueError(f"Default knowledge base directory does not exist: {args.knowledge_base}")
            config_manager.set_override("knowledge_base_path", str(kb_path.resolve()))

        config_manager.set_overrides({
            "embedding_model": args.embedding_model,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k,
            "persist_cache": args.persist_cache,
        })

        logger.info("=== Simplified RAG MCP Server Configuration ===")
        for key, value in config_manager.get_all_values().items():
            logger.info(f"  - {key}: {value}")


# Global server instance using SOLID architecture
server = RAGServer()


# MCP tool definitions - simplified using SOLID architecture
@app.tool()
async def initialize_knowledge_base(
    knowledge_base_path: Union[str, None] = None,
    embedding_model: Union[str, None] = None,
    chunk_size: Union[int, None] = None,
    chunk_overlap: Union[int, None] = None,
    context: Context = None,
) -> str:
    """Initialize a knowledge base from documents (delegates to SOLID architecture)."""
    tool_registry = server.get_tool_registry()
    return await tool_registry.execute_tool(
        KnowledgeBaseInitializeTool.TOOL_NAME,
        knowledge_base_path=knowledge_base_path,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        context=context
    )


@app.tool()
async def refresh_knowledge_base(
    knowledge_base_path: Union[str, None] = None, context: Context = None
) -> str:
    """Refresh the knowledge base with new or changed documents (delegates to SOLID architecture)."""
    tool_registry = server.get_tool_registry()
    return await tool_registry.execute_tool(
        RefreshKnowledgeBaseTool.TOOL_NAME,
        knowledge_base_path=knowledge_base_path,
        context=context
    )


@app.tool()
async def semantic_search(
    query: str,
    knowledge_base_path: Union[str, None] = None,
    top_k: Union[int, None] = None,
    include_scores: bool = False,
) -> str:
    """Perform semantic search on the knowledge base (delegates to SOLID architecture)."""
    tool_registry = server.get_tool_registry()
    return await tool_registry.execute_tool(
        SemanticSearchTool.TOOL_NAME,
        query=query,
        knowledge_base_path=knowledge_base_path,
        top_k=top_k,
        include_scores=include_scores
    )


@app.tool()
async def get_knowledge_base_stats(knowledge_base_path: Union[str, None] = None) -> str:
    """Get statistics about the knowledge base (delegates to SOLID architecture)."""
    tool_registry = server.get_tool_registry()
    return await tool_registry.execute_tool(
        KnowledgeBaseStatsTool.TOOL_NAME,
        knowledge_base_path=knowledge_base_path
    )


@app.tool()
async def list_documents(knowledge_base_path: Union[str, None] = None) -> str:
    """List all documents in the knowledge base (delegates to SOLID architecture)."""
    tool_registry = server.get_tool_registry()
    return await tool_registry.execute_tool(
        ListDocumentsTool.TOOL_NAME,
        knowledge_base_path=knowledge_base_path
    )


async def startup_auto_load():
    """Auto-load cached knowledge base at server startup."""
    config_manager = server.get_config_manager()
    app_state = server.get_app_state()

    kb_path = config_manager.get_all_values().get("knowledge_base_path")
    if not kb_path:
        logger.info("🔄 STARTUP: No default knowledge base path set")
        return

    # Check if cache files exist
    from pathlib import Path
    cache_dir = Path(kb_path) / ".rag_cache"
    if not cache_dir.exists():
        logger.info("🔄 STARTUP: No cache directory found")
        return

    logger.info(f"🚀 STARTUP: Found cache directory for {kb_path}, loading from cache...")
    try:
        # Get KB manager and load from cache
        from .core.service_container import service_container
        from .interfaces.knowledge_base_interfaces import KnowledgeBaseManagerInterface
        kb_manager = service_container.resolve(KnowledgeBaseManagerInterface)

        embedding_model = config_manager.get_embedding_model()
        chunk_size = config_manager.get_chunk_size()
        chunk_overlap = config_manager.get_chunk_overlap()

        documents, faiss_index, embedding_service, document_store = await kb_manager.load_from_cache_only(
            kb_path, embedding_model, chunk_size, chunk_overlap
        )

        # Update application state
        cache_key = f"{kb_path}:{embedding_model}"
        document_processor = kb_manager.get_or_create_document_processor(chunk_size, chunk_overlap)
        app_state.update_knowledge_base_components(
            embedding_service, document_processor, faiss_index, document_store, documents, kb_path, cache_key
        )
        logger.info(f"✅ STARTUP: Successfully loaded {len(documents)} documents from cache")

    except Exception as e:
        logger.warning(f"⚠️ STARTUP: Failed to load from cache: {e}")
        logger.info("🔄 STARTUP: Will initialize knowledge base on first tool use")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RAG MCP Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind the server to")
    parser.add_argument("--knowledge-base", type=str, help="Default knowledge base directory path")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_VALUES["embedding_model"],
        help=f"Default embedding model (default: {DEFAULT_VALUES['embedding_model']})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_VALUES["chunk_size"],
        help=f"Default chunk size for text processing (default: {DEFAULT_VALUES['chunk_size']})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_VALUES["chunk_overlap"],
        help=f"Default chunk overlap (default: {DEFAULT_VALUES['chunk_overlap']})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_VALUES["top_k"],
        help=f"Default number of search results (default: {DEFAULT_VALUES['top_k']})",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--persist-cache",
        action="store_true",
        help="Enable persistence of FAISS index and embeddings cache",
    )
    return parser.parse_args()


def main():
    """Main entry point for the MCP server."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("sentence_transformers").setLevel(logging.DEBUG)

    try:
        # Configure server using SOLID architecture
        server.configure_from_args(args)

        # Auto-load cache at startup if persistence is enabled
        import asyncio
        asyncio.run(startup_auto_load())

        logger.info("=== RAG MCP Server Starting ===")

        if args.port != 8000 or args.host != "localhost":
            logger.info(f"Starting HTTP server on {args.host}:{args.port}")
            app.run_sse(host=args.host, port=args.port)
        else:
            logger.info("Starting stdio server (no port - communicates via stdin/stdout)")
            app.run()

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
