"""
Main MCP server for RAG operations.

Provides tools for knowledge base management and semantic search.
"""

import asyncio
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from .core.document_processor import DocumentProcessor, Document
from .core.embedding_service import EmbeddingService
from .core.faiss_index import FAISSIndex
from .core.document_store import DocumentStore

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
    "max_batch_size": 32
}

print(f"🚀 RAG MCP SERVER - Using hardcoded defaults: {DEFAULT_VALUES}")

# Global state for RAG components
rag_state = {
    "embedding_service": None,
    "document_processor": None,
    "faiss_index": None,
    "document_store": None,
    "documents": [],
    "initialized_kb_paths": set(),
    "default_config": {
        "knowledge_base_path": None,
        **DEFAULT_VALUES  # Spread the default values
    }
}


def get_kb_cache_key(kb_path: str, embedding_model: str) -> str:
    """Generate a cache key for knowledge base configuration."""
    return f"{kb_path}:{embedding_model}"


def resolve_knowledge_base_path(provided_path: str = None) -> str:
    """
    Resolve the knowledge base path using provided path or default.
    
    Args:
        provided_path: Path provided in tool call (optional)
        
    Returns:
        Resolved knowledge base path
        
    Raises:
        ValueError: If no path is provided and no default is set
    """
    if provided_path:
        return provided_path
    
    default_path = rag_state["default_config"]["knowledge_base_path"]
    if default_path:
        return default_path
    
    raise ValueError("No knowledge base path provided and no default path configured. "
                    "Either provide knowledge_base_path in the tool call or start the server with --knowledge-base")


def get_default_value(key: str, provided_value=None):
    """Get default value for configuration parameters from rag_state."""
    if provided_value is not None:
        logger.info(f"Using provided value for {key}: {provided_value}")
        return provided_value
    
    default_val = rag_state["default_config"].get(key)
    logger.info(f"Using default value for {key}: {default_val}")
    return default_val


@app.tool()
async def initialize_knowledge_base(
    knowledge_base_path: str = None,
    embedding_model: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> str:
    """Initialize a knowledge base from documents.
    
    Args:
        knowledge_base_path: Path to directory containing documents (optional if default is set)
        embedding_model: SentenceTransformer model name for embeddings (default: ibm-granite/granite-embedding-278m-multilingual)
        chunk_size: Maximum size of text chunks in characters (default: 500)
        chunk_overlap: Overlap between chunks in characters (default: 200)
        
    Returns:
        Initialization status and statistics
    """
    try:
        # Resolve parameters using defaults
        kb_path_str = resolve_knowledge_base_path(knowledge_base_path)
        embedding_model = get_default_value("embedding_model", embedding_model)
        chunk_size = get_default_value("chunk_size", chunk_size)
        chunk_overlap = get_default_value("chunk_overlap", chunk_overlap)
        
        logger.info(f"=== Knowledge Base Initialization ===")
        logger.info(f"Path: {kb_path_str}")
        logger.info(f"Embedding model: {embedding_model}")
        logger.info(f"Chunk size: {chunk_size}")
        logger.info(f"Chunk overlap: {chunk_overlap}")
        
        kb_path = Path(kb_path_str)
        if not kb_path.exists():
            raise ValueError(f"Knowledge base directory does not exist: {kb_path_str}")
        
        # Initialize components with the resolved values
        logger.info(f"Initializing EmbeddingService with model: {embedding_model}")
        embedding_service = EmbeddingService(embedding_model)
        
        logger.info(f"Initializing DocumentProcessor with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        
        faiss_index = FAISSIndex(embedding_service.dimension)
        
        # Create document store in the same directory as the knowledge base
        store_path = kb_path / "document_store.db"
        document_store = DocumentStore(str(store_path))
        
        # Load and process documents
        logger.info("Loading and processing documents...")
        documents = document_processor.load_documents(kb_path)
        
        if not documents:
            return f"No documents found in {kb_path_str}. Supported extensions: {document_processor.get_supported_extensions()}"
        
        logger.info(f"Processed into {len(documents)} chunks (chunk_size={chunk_size})")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [doc.content for doc in documents]
        embeddings = embedding_service.get_embeddings(texts)
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        faiss_index.add_embeddings(embeddings)
        
        # Update document store
        logger.info("Updating document store...")
        filename_chunks = {}
        for doc in documents:
            filename_chunks[doc.filename] = filename_chunks.get(doc.filename, 0) + 1
        
        files = [f for f in kb_path.iterdir() 
                if f.is_file() and f.suffix.lower() in document_processor.get_supported_extensions()]
        
        for file_path in files:
            if file_path.name in filename_chunks:
                file_hash = document_store.compute_file_hash(file_path)
                chunk_count = filename_chunks[file_path.name]
                document_store.store_document(file_path, file_hash, chunk_count)
        
        # Update global state
        cache_key = get_kb_cache_key(kb_path_str, embedding_model)
        rag_state.update({
            "embedding_service": embedding_service,
            "document_processor": document_processor,
            "faiss_index": faiss_index,
            "document_store": document_store,
            "documents": documents,
            "current_kb_path": kb_path_str,
            "current_cache_key": cache_key
        })
        rag_state["initialized_kb_paths"].add(cache_key)
        
        # Get model info for verification
        model_info = embedding_service.get_model_info()
        actual_model_used = model_info.get('actual_model_path', model_info.get('model_name'))
        
        result_message = f"""Knowledge base initialized successfully!
- Path: {kb_path_str}
- Documents: {len(filename_chunks)} files
- Chunks: {len(documents)} (chunk_size: {chunk_size}, overlap: {chunk_overlap})
- Embedding model: {actual_model_used}
- Embedding dimension: {embedding_service.dimension}"""

        if model_info.get('model_mismatch'):
            result_message += f"\n⚠️  WARNING: Model mismatch detected"
        
        logger.info("Knowledge base initialization completed successfully!")
        return result_message
        
    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {e}")
        raise


@app.tool()
async def refresh_knowledge_base(knowledge_base_path: str = None) -> str:
    """Refresh the knowledge base with new or changed documents."""
    try:
        kb_path_str = resolve_knowledge_base_path(knowledge_base_path)
        if not rag_state.get("current_kb_path") == kb_path_str:
            return f"Knowledge base not initialized for path: {kb_path_str}. Please initialize first."
        
        kb_path = Path(kb_path_str)
        if not kb_path.exists():
            raise ValueError(f"Knowledge base directory does not exist: {kb_path_str}")
        
        document_processor = rag_state["document_processor"]
        document_store = rag_state["document_store"]
        embedding_service = rag_state["embedding_service"]
        
        logger.info(f"Refreshing with current chunk settings: size={document_processor.chunk_size}, overlap={document_processor.chunk_overlap}")
        
        extensions = document_processor.get_supported_extensions()
        current_files = [f for f in kb_path.iterdir() 
                        if f.is_file() and f.suffix.lower() in extensions]
        current_file_names = {f.name for f in current_files}
        
        stored_file_names = document_store.get_all_document_names()
        removed_files = stored_file_names - current_file_names
        
        new_or_changed_files = []
        for file_path in current_files:
            has_changed, file_hash = document_store.is_document_changed(file_path)
            if has_changed:
                new_or_changed_files.append((file_path, file_hash))
        
        if not new_or_changed_files and not removed_files:
            return "No changes detected in knowledge base."
        
        for filename in removed_files:
            document_store.remove_document(filename)
        
        if removed_files or new_or_changed_files:
            documents = document_processor.load_documents(kb_path)
            faiss_index = FAISSIndex(embedding_service.dimension)
            
            if documents:
                texts = [doc.content for doc in documents]
                embeddings = embedding_service.get_embeddings(texts)
                faiss_index.add_embeddings(embeddings)
                
                filename_chunks = {}
                for doc in documents:
                    filename_chunks[doc.filename] = filename_chunks.get(doc.filename, 0) + 1
                
                for file_path, file_hash in new_or_changed_files:
                    if file_path.name in filename_chunks:
                        chunk_count = filename_chunks[file_path.name]
                        document_store.store_document(file_path, file_hash, chunk_count)
            
            rag_state.update({
                "faiss_index": faiss_index,
                "documents": documents
            })
            
            return f"""Knowledge base refreshed successfully!
- Removed files: {len(removed_files)}
- New/changed files: {len(new_or_changed_files)}
- Total documents: {len(set(doc.filename for doc in documents))}
- Total chunks: {len(documents)} (with current chunk settings)"""
        
    except Exception as e:
        logger.error(f"Failed to refresh knowledge base: {e}")
        raise


@app.tool()
async def semantic_search(
    query: str,
    knowledge_base_path: str = None,
    top_k: int = None,
    include_scores: bool = False
) -> str:
    """Perform semantic search on the knowledge base."""
    try:
        kb_path_str = resolve_knowledge_base_path(knowledge_base_path)
        
        if top_k is None:
            top_k = get_default_value("top_k")
        
        if not rag_state.get("current_kb_path") == kb_path_str:
            return f"Knowledge base not initialized for path: {kb_path_str}. Please initialize first."
        
        embedding_service = rag_state["embedding_service"]
        faiss_index = rag_state["faiss_index"]
        documents = rag_state["documents"]
        
        if not faiss_index.initialized:
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
                    "chunk_index": doc.chunk_index,
                    "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
                }
                if include_scores:
                    result["similarity_score"] = float(distance)
                
                results.append(result)
        
        if not results:
            return "No relevant documents found for the query."
        
        response_text = f"Found {len(results)} relevant document chunks for query: '{query}' (top_k={top_k})\n\n"
        
        for result in results:
            response_text += f"**Rank {result['rank']}** - {result['filename']} (chunk {result['chunk_index']})\n"
            if include_scores:
                response_text += f"Similarity Score: {result['similarity_score']:.4f}\n"
            response_text += f"Content: {result['content']}\n\n"
        
        return response_text
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise


@app.tool()
async def get_knowledge_base_stats(knowledge_base_path: str = None) -> str:
    """Get statistics about the knowledge base."""
    try:
        kb_path_str = resolve_knowledge_base_path(knowledge_base_path)
        
        if not rag_state.get("current_kb_path") == kb_path_str:
            return f"Knowledge base not initialized for path: {kb_path_str}. Please initialize first."
        
        embedding_service = rag_state["embedding_service"]
        faiss_index = rag_state["faiss_index"]
        document_store = rag_state["document_store"]
        documents = rag_state["documents"]
        document_processor = rag_state["document_processor"]
        
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
- Default top_k: {rag_state['default_config']['top_k']}

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


@app.tool()
async def list_documents(knowledge_base_path: str = None) -> str:
    """List all documents in the knowledge base."""
    try:
        kb_path_str = resolve_knowledge_base_path(knowledge_base_path)
        
        if not rag_state.get("current_kb_path") == kb_path_str:
            return f"Knowledge base not initialized for path: {kb_path_str}. Please initialize first."
        
        documents = rag_state["documents"]
        document_store = rag_state["document_store"]
        
        files_info = {}
        for doc in documents:
            if doc.filename not in files_info:
                files_info[doc.filename] = {
                    "chunk_count": 0,
                    "total_content_length": 0
                }
            files_info[doc.filename]["chunk_count"] += 1
            files_info[doc.filename]["total_content_length"] += len(doc.content)
        
        if not files_info:
            return "No documents found in the knowledge base."
        
        response_text = f"Documents in knowledge base: {kb_path_str}\n\n"
        
        for filename in sorted(files_info.keys()):
            info = files_info[filename]
            doc_info = document_store.get_document_info(filename)
            
            response_text += f"**{filename}**\n"
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RAG MCP Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind the server to")
    parser.add_argument("--knowledge-base", type=str, help="Default knowledge base directory path")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_VALUES["embedding_model"], 
                       help=f"Default embedding model (default: {DEFAULT_VALUES['embedding_model']})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_VALUES["chunk_size"], 
                       help=f"Default chunk size for text processing (default: {DEFAULT_VALUES['chunk_size']})")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_VALUES["chunk_overlap"], 
                       help=f"Default chunk overlap (default: {DEFAULT_VALUES['chunk_overlap']})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_VALUES["top_k"], 
                       help=f"Default number of search results (default: {DEFAULT_VALUES['top_k']})")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main():
    """Main entry point for the MCP server."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("sentence_transformers").setLevel(logging.DEBUG)
    
    if args.knowledge_base:
        kb_path = Path(args.knowledge_base)
        if not kb_path.exists():
            logger.error(f"Default knowledge base directory does not exist: {args.knowledge_base}")
            return 1
        rag_state["default_config"]["knowledge_base_path"] = str(kb_path.resolve())
    
    # Override defaults with command line arguments if provided
    if args.embedding_model != DEFAULT_VALUES["embedding_model"]:
        rag_state["default_config"]["embedding_model"] = args.embedding_model
        logger.info(f"Command line override: embedding_model = {args.embedding_model}")
    
    if args.chunk_size != DEFAULT_VALUES["chunk_size"]:
        rag_state["default_config"]["chunk_size"] = args.chunk_size
        logger.info(f"Command line override: chunk_size = {args.chunk_size}")
    
    if args.chunk_overlap != DEFAULT_VALUES["chunk_overlap"]:
        rag_state["default_config"]["chunk_overlap"] = args.chunk_overlap
        logger.info(f"Command line override: chunk_overlap = {args.chunk_overlap}")
    
    if args.top_k != DEFAULT_VALUES["top_k"]:
        rag_state["default_config"]["top_k"] = args.top_k
        logger.info(f"Command line override: top_k = {args.top_k}")
    
    logger.info(f"=== RAG MCP Server Starting ===")
    logger.info(f"Final configuration:")
    for key, value in rag_state["default_config"].items():
        logger.info(f"  - {key}: {value}")
    
    if args.port != 8000 or args.host != "localhost":
        logger.info(f"Starting HTTP server on {args.host}:{args.port}")
        app.run_sse(host=args.host, port=args.port)
    else:
        logger.info("Starting stdio server (no port - communicates via stdin/stdout)")
        app.run()


if __name__ == "__main__":
    main()
