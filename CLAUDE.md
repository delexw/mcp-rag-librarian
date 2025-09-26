# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) MCP (Model Context Protocol) server that provides tools for building and querying vector-based knowledge bases from document collections. The server enables semantic search and document retrieval capabilities through a FastMCP server implementation.

## Architecture

The codebase follows a clean modular architecture:

- **`src/rag_mcp_server/server.py`**: Main MCP server implementation with FastMCP, handles all tool endpoints and global state management
- **`src/rag_mcp_server/core/`**: Core functionality modules:
  - `document_processor.py`: Document loading, text extraction, and chunking (supports .txt, .pdf)
  - `embedding_service.py`: Text embedding generation using SentenceTransformers
  - `faiss_index.py`: Vector similarity search using FAISS IndexFlatIP
  - `document_store.py`: SQLite-based document metadata and change tracking

Key architectural patterns:
- **Global State Management**: `rag_state` dictionary maintains all components and configuration
- **Cache Key System**: Knowledge bases are cached by `path:embedding_model` combinations
- **Incremental Processing**: Only processes new/changed documents using file hash tracking
- **Default Value Resolution**: Configuration cascades from hardcoded defaults → CLI args → tool parameters

## Development Commands

### Installation and Setup
```bash
# Development installation from source
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Code Quality and Testing
```bash
# Format code
black src/
isort src/

# Type checking
mypy src/

# Run tests
pytest tests/
```

### Running the Server

**Standard MCP mode (stdio):**
```bash
# Basic usage
python -m rag_mcp_server.server

# With configuration
python -m rag_mcp_server.server --knowledge-base /path/to/docs --embedding-model "all-MiniLM-L6-v2" --chunk-size 800 --verbose
```

**HTTP mode (for testing):**
```bash
# HTTP server mode
python -m rag_mcp_server.server --host 0.0.0.0 --port 8080
```

**Using uvx (recommended for distribution):**
```bash
# Run from source with uvx
uvx --from . rag-mcp-server --knowledge-base ./test_documents --verbose
```

### Development Configuration

The server uses hardcoded defaults that can be overridden:
```python
DEFAULT_VALUES = {
    "embedding_model": "ibm-granite/granite-embedding-278m-multilingual",
    "chunk_size": 500,
    "chunk_overlap": 200,
    "top_k": 7,
    "max_batch_size": 32
}
```

## MCP Tools Available

1. **`initialize_knowledge_base`**: Initialize knowledge base from document directory
2. **`semantic_search`**: Perform semantic search with configurable top_k and scoring
3. **`refresh_knowledge_base`**: Update with new/changed documents (incremental)
4. **`get_knowledge_base_stats`**: Get detailed statistics and configuration info
5. **`list_documents`**: List all processed documents with metadata

## Key Implementation Details

### Document Processing Pipeline
1. File detection (`.txt`, `.pdf` extensions)
2. Text extraction with encoding fallback (UTF-8 → Latin-1)
3. Intelligent chunking with configurable size/overlap while preserving word boundaries
4. Batch embedding generation with progress tracking
5. FAISS index building (IndexFlatIP with cosine similarity via L2 normalization)

### State Management
- Knowledge bases are cached by `get_kb_cache_key(kb_path, embedding_model)`
- Global `rag_state` maintains current components and configuration
- Document store tracks file hashes for incremental updates
- Error handling includes model fallback and transaction rollback

### Configuration Resolution Order
1. Hardcoded `DEFAULT_VALUES`
2. Command line arguments override defaults
3. Tool call parameters override both (highest priority)

The `get_default_value()` and `resolve_knowledge_base_path()` functions handle this cascading configuration system.