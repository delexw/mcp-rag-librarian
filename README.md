# MCP RAG Librarian 📚🦸‍♀️

<div align="center">
  <h1>MCP RAG Librarian</h1>
  <p>
    <a href="https://pypi.org/project/mcp-rag-librarian/"><img src="https://img.shields.io/pypi/v/mcp-rag-librarian" alt="PyPI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/github/license/delexw/mcp-rag-librarian" alt="License"></a>
  </p>
  <p><strong>Fork of <a href="https://github.com/tungetti/rag-mcp-server">tungetti/rag-mcp-server</a> with persistence enhancements</strong></p>
  <p><em>Original work by <a href="https://github.com/tungetti">Tommaso Maria Ungetti</a></em></p>
</div>

Your superhero librarian for intelligent document retrieval! 🦸‍♀️📚

A Model Context Protocol (MCP) server for Retrieval-Augmented Generation (RAG) operations. RAG Librarian provides tools for building and querying vector-based knowledge bases from document collections, enabling semantic search and document retrieval capabilities with the power of a superhero librarian!

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Setup](#setup)
  - [Find the MCP settings file for the client](#find-the-mcp-settings-file-for-the-client)
    - [Claude Desktop](#claude-desktop)
    - [Claude Code](#claude-code)
    - [Cursor](#cursor)
    - [Cline](#cline)
    - [Windsurf](#windsurf)
    - [Any other client](#any-other-client)
  - [Set up the MCP server](#set-up-the-mcp-server)
  - [Variant: Manual setup with uvx](#variant-manual-setup-with-uvx)
- [Usage Examples](#usage-examples)
  - [Sample LLM Queries](#sample-llm-queries)
  - [Command Line Examples](#command-line-examples)
- [MCP Tools](#mcp-tools)
- [Technical Details](#technical-details)
- [Configuration Examples](#configuration-examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Document Processing**: Supports multiple file formats (.txt, .pdf, .md) with automatic text extraction and recursive directory scanning
- **Intelligent Chunking**: Configurable text chunking with overlap to preserve context
- **Vector Embeddings**: Uses SentenceTransformers for high-quality text embeddings
- **Semantic Search**: FAISS-powered similarity search for fast and accurate retrieval
- **Incremental Updates**: Smart document tracking to only process new or changed files
- **Persistent Caching**: FAISS index and embeddings persistence with `--persist-cache` flag
- **Auto-Load on Startup**: Cached knowledge bases load automatically when server starts
- **Progress Notifications**: Real-time progress updates during long-running operations via MCP progress notifications
- **SOLID Architecture**: Clean, extensible persistence layer following SOLID principles
- **SQLite Document Store**: Document metadata and change tracking
- **Flexible Configuration**: Customizable embedding models, chunk sizes, and search parameters

## Architecture

```
mcp-rag-librarian/
├── src/rag_mcp_server/
│   ├── server.py              # Main MCP server implementation
│   └── core/
│       ├── document_processor.py   # Document loading and chunking
│       ├── embedding_service.py    # Text embedding generation
│       ├── faiss_index.py          # Vector similarity search
│       ├── document_store.py       # Document metadata storage
│       ├── persistence.py          # SOLID persistence abstractions
│       ├── file_persistence.py     # File-based persistence provider
│       └── persistence_factory.py  # Dependency injection factory
```

## Installation

### Using uvx (Recommended)

```bash
# Install with uvx (comes with uv)
uvx mcp-rag-librarian
```

### Using pip

```bash
pip install mcp-rag-librarian
```

### From source

```bash
git clone <repository-url>
cd mcp-rag-librarian
pip install -e .
```

## Setup

The easiest way to run the MCP server is with `uvx`, but manual setup is also available.

### Find the MCP settings file for the client

#### Claude Desktop

1. [Install Claude Desktop](https://claude.ai/download) as needed
2. Open the config file by opening the Claude Desktop app, going into its Settings, opening the 'Developer' tab, and clicking the 'Edit Config' button
3. Follow the 'Set up the MCP server' steps below

#### Claude Code

1. Install [Claude Code](https://docs.anthropic.com/en/docs/claude-code/getting-started) as needed
2. Run the following command to add the RAG server:

   ```bash
   claude mcp add rag
   ```

   Or manually add with custom configuration:

   ```bash
   claude mcp add-json rag '{"command":"uvx","args":["mcp-rag-librarian","--knowledge-base","/path/to/your/docs","--embedding-model","all-MiniLM-L6-v2","--chunk-size","1000","--chunk-overlap","200"]}'
   ```

#### Cursor

1. [Install Cursor](https://www.cursor.com/downloads) as needed
2. Open the config file by opening Cursor, going into 'Cursor Settings' (not the normal VSCode IDE settings), opening the 'MCP' tab, and clicking the 'Add new global MCP server' button
3. Follow the 'Set up the MCP server' steps below

#### Cline

1. [Install Cline](https://cline.bot/) in your IDE as needed
2. Open the config file by opening your IDE, opening the Cline sidebar, clicking the 'MCP Servers' icon button that is second from left at the top, opening the 'Installed' tab, and clicking the 'Configure MCP Servers' button
3. Follow the 'Set up the MCP server' steps below

#### Windsurf

1. [Install Windsurf](https://windsurf.com/download) as needed
2. Open the config file by opening Windsurf, going into 'Windsurf Settings' (not the normal VSCode IDE settings), opening the 'Cascade' tab, and clicking the 'View raw config' button in the 'Model Context Protocol (MCP) Servers' section
3. Follow the 'Set up the MCP server' steps below

#### Any other client

1. Find the MCP settings file, usually something like `[client]_mcp_config.json`
2. Follow the 'Set up the MCP server' steps below

### Set up the MCP server

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) as needed (uvx comes bundled with uv)
2. Add the following to your MCP setup:

   **Basic Configuration:**
   ```json
   {
     "mcpServers": {
       "rag": {
         "command": "uvx",
         "args": ["mcp-rag-librarian"]
       }
     }
   }
   ```

   **Full Configuration with All Parameters (including persistence):**
   ```json
   {
     "mcpServers": {
       "rag": {
         "command": "uvx",
         "args": [
           "mcp-rag-librarian",
           "--knowledge-base", "/path/to/your/documents",
           "--embedding-model", "ibm-granite/granite-embedding-278m-multilingual",
           "--chunk-size", "500",
           "--chunk-overlap", "200",
           "--top-k", "7",
           "--persist-cache",
           "--verbose"
         ]
       }
     }
   }
   ```

### Variant: Manual setup with uvx

If you prefer to run the server manually or need specific Python version:

```bash
# Run with default settings
uvx mcp-rag-librarian

# Run with all parameters specified (including persistence)
uvx mcp-rag-librarian \
  --knowledge-base /path/to/documents \
  --embedding-model "ibm-granite/granite-embedding-278m-multilingual" \
  --chunk-size 500 \
  --chunk-overlap 200 \
  --top-k 7 \
  --persist-cache \
  --verbose

# Run from source directory with persistence
uvx --from . mcp-rag-librarian \
  --knowledge-base /home/user/documents \
  --embedding-model "all-MiniLM-L6-v2" \
  --chunk-size 800 \
  --chunk-overlap 100 \
  --top-k 5 \
  --persist-cache
```

## Persistence Feature

**🆕 NEW**: This fork adds persistent caching capabilities to avoid re-processing documents on server restart.

### Key Benefits
- **Faster Startup**: Skip re-initialization when cache exists
- **Automatic Loading**: Cached knowledge bases load on server startup
- **Smart Caching**: Uses MD5 hash of configuration to ensure cache consistency
- **SOLID Architecture**: Extensible persistence layer with clean abstractions

### How to Enable Persistence

Add the `--persist-cache` flag to your configuration:

```json
{
  "mcpServers": {
    "rag": {
      "command": "uvx",
      "args": [
        "mcp-rag-librarian",
        "--knowledge-base", "/path/to/your/docs",
        "--persist-cache"
      ]
    }
  }
}
```

### Cache Behavior
- **Cache Location**: `.rag_cache/` directory alongside your knowledge base
- **Cache Key**: Based on path + embedding model + chunk settings
- **Auto-Load**: Cached data loads automatically when server starts
- **Fallback**: Falls back to normal initialization if cache is invalid

### Example Workflow
1. **First Run**: Initialize knowledge base normally (creates cache)
2. **Server Restart**: Cached data loads automatically at startup
3. **Ready to Search**: No manual initialization needed

## Progress Notifications

**🆕 NEW**: The server now supports **MCP progress notifications** for real-time updates during long-running operations.

### How Progress Notifications Work

When a client (like Claude Desktop or Claude Code) supports MCP progress notifications, the server will send real-time progress updates during:

- **Knowledge Base Initialization**: Progress through document loading, embedding generation, and index building
- **Knowledge Base Refresh**: Progress through change detection, document processing, and index updates

### Progress Notification Features

- **Real-Time Updates**: See progress as operations happen, not just at completion
- **Detailed Messages**: Descriptive progress messages (e.g., "Loading documents...", "Generating embeddings for 1,247 chunks...")
- **Percentage Tracking**: Progress bars with completion percentages (0-100%)
- **Automatic Detection**: Works automatically when client supports progress notifications

### Example Progress Flow

**During Knowledge Base Initialization:**
```
0%   - "Starting knowledge base initialization..."
5%   - "Initializing persistence strategy..."
10%  - "Loading knowledge base from cache or creating new..."
15%  - "Initializing embedding service..."
25%  - "Loading and processing documents..."
40%  - "Generating embeddings for 1,247 chunks..."
70%  - "Building FAISS search index..."
75%  - "Setting up document store and finalizing..."
90%  - "Updating global state..."
100% - "Knowledge base initialization complete!"
```

**During Knowledge Base Refresh:**
```
0%   - "Starting knowledge base refresh..."
10%  - "Scanning for file changes..."
25%  - "Detecting changes in existing files..."
30%  - "Processing 3 removed files..."
40%  - "Reprocessing 5 changed documents..."
60%  - "Regenerating embeddings for 234 chunks..."
80%  - "Rebuilding search index..."
95%  - "Updating global state..."
100% - "Knowledge base refresh complete!"
```

### Client Support

Progress notifications work automatically with MCP clients that support the progress notification protocol:

- ✅ **Claude Code**: Full progress notification support
- ✅ **Claude Desktop**: Full progress notification support
- ✅ **Cursor**: Full progress notification support
- ✅ **Cline**: Full progress notification support
- ✅ **Windsurf**: Full progress notification support

### Technical Details

Progress notifications use the MCP progress notification protocol:
- **Protocol**: JSON-RPC 2.0 notifications with `notifications/progress` method
- **Token-Based**: Each operation gets a unique progress token
- **Optional**: Gracefully degrades when client doesn't support progress notifications
- **Rate-Limited**: Progress updates are sent at reasonable intervals

### Disabling Progress Notifications

Progress notifications are automatically enabled when supported by the client. They cannot be disabled, but if your client doesn't support progress notifications, no extra overhead is incurred.

## Usage Examples

### Sample LLM Queries

Here are example queries you can use with your LLM to interact with the RAG server:

**Initialize a knowledge base with custom parameters:**
```
Initialize the knowledge base with:
- knowledge_base_path: "/home/user/research_papers"
- embedding_model: "ibm-granite/granite-embedding-278m-multilingual"
- chunk_size: 300
- chunk_overlap: 50
```

**Search with specific parameters:**
```
Search for "machine learning optimization techniques" in the knowledge base at "/home/user/research_papers" and return the top 10 results with similarity scores.
```

**Initialize with high-quality embeddings:**
```
Set up a knowledge base at "/data/technical_docs" using the "all-mpnet-base-v2" model with chunk_size of 1000 and chunk_overlap of 400 for better context preservation.
```

**Refresh and get statistics:**
```
Refresh the knowledge base at "/home/user/documents" to include any new files, then show me the statistics including total documents, chunks, and current configuration.
```

**List and search documents:**
```
List all documents in the knowledge base, then search for information about "API authentication" and show me the top 5 most relevant chunks.
```

**Complex workflow example:**
```
1. Initialize a knowledge base at "/home/user/project_docs" with embedding_model "all-MiniLM-L6-v2", chunk_size 800, and chunk_overlap 150
2. Show me the statistics
3. Search for "database optimization strategies"
4. List all documents that were processed
```

**Multilingual search example:**
```
Initialize the knowledge base at "/docs/international" using the multilingual model "ibm-granite/granite-embedding-278m-multilingual", then search for "machine learning" in multiple languages and show the top 7 results.
```

### Command Line Examples

**High-Quality Configuration for Research:**
```bash
uvx mcp-rag-librarian \
  --knowledge-base /home/tommasomariaungetti/RAG \
  --embedding-model "all-mpnet-base-v2" \
  --chunk-size 1000 \
  --chunk-overlap 400 \
  --top-k 10 \
  --verbose
```

**Fast Processing for Large Document Sets:**
```bash
uvx mcp-rag-librarian \
  --knowledge-base /data/large_corpus \
  --embedding-model "all-MiniLM-L6-v2" \
  --chunk-size 2000 \
  --chunk-overlap 100 \
  --top-k 5
```

**Multilingual Document Processing:**
```bash
uvx mcp-rag-librarian \
  --knowledge-base /docs/multilingual \
  --embedding-model "ibm-granite/granite-embedding-278m-multilingual" \
  --chunk-size 500 \
  --chunk-overlap 200 \
  --top-k 7
```

**Running from Source with Custom Settings:**
```bash
uvx --from . mcp-rag-librarian \
  --embedding-model "all-MiniLM-L6-v2" \
  --chunk-size 800 \
  --chunk-overlap 100 \
  --top-k 5 \
  --knowledge-base /home/tommasomariaungetti/RAG
```

## MCP Tools

The following tools are available:

### 1. initialize_knowledge_base

Initialize a knowledge base from a directory of documents with real-time progress notifications.

**Parameters:**
- `knowledge_base_path` (optional): Path to document directory - defaults to server config
- `embedding_model` (optional): Model name for embeddings - defaults to "ibm-granite/granite-embedding-278m-multilingual"
- `chunk_size` (optional): Maximum chunk size in characters - defaults to 500
- `chunk_overlap` (optional): Chunk overlap size in characters - defaults to 200

**Progress Notifications:** ✅ Provides real-time progress updates during document loading, embedding generation, and index building.

**Example Tool Call:**
```json
{
  "tool": "initialize_knowledge_base",
  "arguments": {
    "knowledge_base_path": "/path/to/docs",
    "embedding_model": "all-mpnet-base-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200
  }
}
```

**Example LLM Query:**
> "Initialize a knowledge base from /home/user/documents using the all-mpnet-base-v2 embedding model with 1000 character chunks and 200 character overlap"

### 2. semantic_search

Perform semantic search on the knowledge base.

**Parameters:**
- `query`: Search query text
- `knowledge_base_path` (optional): Path to knowledge base - defaults to current KB
- `top_k` (optional): Number of results to return - defaults to 7
- `include_scores` (optional): Include similarity scores - defaults to false

**Example Tool Call:**
```json
{
  "tool": "semantic_search",
  "arguments": {
    "query": "How to implement RAG systems?",
    "knowledge_base_path": "/path/to/docs",
    "top_k": 5,
    "include_scores": true
  }
}
```

**Example LLM Query:**
> "Search for 'machine learning optimization techniques' and show me the top 5 results with similarity scores"

### 3. refresh_knowledge_base

Update the knowledge base with new or changed documents with real-time progress notifications.

**Parameters:**
- `knowledge_base_path` (optional): Path to knowledge base - defaults to current KB

**Progress Notifications:** ✅ Provides real-time progress updates during change detection, document reprocessing, and index rebuilding.

**Example Tool Call:**
```json
{
  "tool": "refresh_knowledge_base",
  "arguments": {
    "knowledge_base_path": "/path/to/docs"
  }
}
```

**Example LLM Query:**
> "Refresh the knowledge base to include any new or modified documents"

### 4. get_knowledge_base_stats

Get detailed statistics about the knowledge base.

**Parameters:**
- `knowledge_base_path` (optional): Path to knowledge base - defaults to current KB

**Example Tool Call:**
```json
{
  "tool": "get_knowledge_base_stats",
  "arguments": {
    "knowledge_base_path": "/path/to/docs"
  }
}
```

**Example LLM Query:**
> "Show me the statistics for the knowledge base including document count, chunk information, and current configuration"

### 5. list_documents

List all documents in the knowledge base with metadata.

**Parameters:**
- `knowledge_base_path` (optional): Path to knowledge base - defaults to current KB

**Example Tool Call:**
```json
{
  "tool": "list_documents",
  "arguments": {
    "knowledge_base_path": "/path/to/docs"
  }
}
```

**Example LLM Query:**
> "List all documents in the knowledge base with their chunk counts and metadata"

## Technical Details

### Document Processing

The system uses a sophisticated document processing pipeline:

1. **File Discovery**: Recursively scans directories and subdirectories for supported file types
2. **Supported Formats**:
   - `.txt` files: Plain text documents
   - `.pdf` files: PDF documents with text extraction
   - `.md` files: Markdown documents (processed as text)
3. **Content Extraction**:
   - Plain text/Markdown: Direct UTF-8/Latin-1 reading with encoding fallback
   - PDF files: PyMuPDF-based text extraction
4. **Text Chunking**:
   - Splits documents into manageable chunks
   - Preserves word boundaries
   - Maintains context with configurable overlap

### Embedding Generation

- **Default Model**: `ibm-granite/granite-embedding-278m-multilingual`
- **Batch Processing**: Efficient batch encoding for large document sets
- **Fallback Support**: Automatic fallback to `all-MiniLM-L6-v2` if primary model fails
- **Progress Tracking**: Visual progress bars for large operations

### Vector Search

- **Index Type**: FAISS IndexFlatIP (Inner Product)
- **Similarity Metric**: Cosine similarity (via L2 normalization)
- **Performance**: Scales to millions of documents
- **Accuracy**: Exact nearest neighbor search

### Document Store

- **Storage**: SQLite database
- **Tracking**: File hash, modification time, chunk count
- **Incremental Updates**: Only processes changed files
- **Location**: Stored alongside knowledge base documents

## Configuration Examples

### MCP Client Configurations

**Basic Configuration (Claude Desktop/Cursor/Cline):**
```json
{
  "mcpServers": {
    "rag": {
      "command": "uvx",
      "args": ["mcp-rag-librarian"]
    }
  }
}
```

**Full Configuration with All Parameters:**
```json
{
  "mcpServers": {
    "rag": {
      "command": "uvx",
      "args": [
        "mcp-rag-librarian",
        "--knowledge-base", "/path/to/documents",
        "--embedding-model", "ibm-granite/granite-embedding-278m-multilingual",
        "--chunk-size", "500",
        "--chunk-overlap", "200",
        "--top-k", "7",
        "--verbose"
      ]
    }
  }
}
```

**Multiple Knowledge Base Configuration:**
```json
{
  "mcpServers": {
    "rag-technical": {
      "command": "uvx",
      "args": [
        "mcp-rag-librarian",
        "--knowledge-base", "/docs/technical",
        "--embedding-model", "all-mpnet-base-v2",
        "--chunk-size", "1000",
        "--chunk-overlap", "400"
      ]
    },
    "rag-research": {
      "command": "uvx",
      "args": [
        "mcp-rag-librarian",
        "--knowledge-base", "/docs/research",
        "--embedding-model", "all-MiniLM-L6-v2",
        "--chunk-size", "500",
        "--chunk-overlap", "100",
        "--port", "8001"
      ]
    }
  }
}
```

### Command Line Examples

**High-Quality Configuration for Research:**
```bash
uvx mcp-rag-librarian \
  --knowledge-base /path/to/research/docs \
  --embedding-model "all-mpnet-base-v2" \
  --chunk-size 1000 \
  --chunk-overlap 400 \
  --top-k 10
```

**Fast Processing Configuration:**
```bash
uvx mcp-rag-librarian \
  --knowledge-base /path/to/large/corpus \
  --embedding-model "all-MiniLM-L6-v2" \
  --chunk-size 2000 \
  --chunk-overlap 100 \
  --top-k 5
```

**Multilingual Configuration:**
```bash
uvx mcp-rag-librarian \
  --knowledge-base /path/to/multilingual/docs \
  --embedding-model "ibm-granite/granite-embedding-278m-multilingual" \
  --chunk-size 500 \
  --chunk-overlap 200 \
  --top-k 7
```

**Development Configuration with Verbose Logging:**
```bash
uvx --from . mcp-rag-librarian \
  --knowledge-base ./test_documents \
  --embedding-model "all-MiniLM-L6-v2" \
  --chunk-size 300 \
  --chunk-overlap 50 \
  --top-k 3 \
  --verbose
```

## Error Handling

The server implements comprehensive error handling:

- **File Access Errors**: Graceful handling of permission issues
- **Encoding Errors**: Automatic encoding detection and fallback
- **Model Loading Errors**: Fallback to default models
- **Database Errors**: Transaction rollback and recovery
- **Search Errors**: Informative error messages

## Performance Considerations

### Memory Usage
- Embeddings are stored in memory for fast search
- Approximate memory: `num_chunks × embedding_dimension × 4 bytes`
- Example: 10,000 chunks × 384 dimensions ≈ 15 MB

### Processing Speed
- Document processing: ~100-500 docs/minute (depending on size)
- Embedding generation: ~50-200 chunks/second (model dependent)
- Search latency: <10ms for 100K documents

### Optimization Tips
1. Use smaller embedding models for faster processing
2. Increase chunk size for fewer chunks (may reduce accuracy)
3. Decrease overlap for faster processing (may lose context)
4. Use SSD storage for document store database

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
isort src/
```

### Type Checking
```bash
mypy src/
```

## Troubleshooting

### Common Issues

1. **"No knowledge base path provided"**
   - Solution: Either provide path in tool call or use `--knowledge-base` flag

2. **"Model mismatch detected"**
   - Solution: This is a warning; the system will use the closest available model

3. **"Failed to initialize embedding model"**
   - Solution: Check internet connection or use a locally cached model

4. **"No documents found in knowledge base"**
   - Solution: Ensure directory contains .txt, .pdf, or .md files (searches recursively in subdirectories)

### Debug Mode

Enable verbose logging for troubleshooting:
```bash
uvx mcp-rag-librarian --verbose
```

### Testing with MCP Inspector

Test your RAG server interactively using MCP Inspector:

```bash
npx @modelcontextprotocol/inspector --config mcp_inspector_config.json
```

This will launch a web interface where you can test all MCP tools interactively. Make sure you have a `mcp_inspector_config.json` file in your project root with your server configuration.

## Help and Resources

- [GitHub Repository](https://github.com/delexw/mcp-rag-librarian)
- [PyPI Package](https://pypi.org/project/mcp-rag-librarian/)
- [Original Repository](https://github.com/tungetti/rag-mcp-server)
- [MCP Documentation](https://github.com/modelcontextprotocol/mcp)
- [Issue Tracker](https://github.com/delexw/mcp-rag-librarian/issues)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **Original Work**: This is a fork of [rag-mcp-server](https://github.com/tungetti/rag-mcp-server) by [Tommaso Maria Ungetti](https://github.com/tungetti)
- Built on [MCP (Model Context Protocol)](https://github.com/anthropics/mcp)
- Powered by [Sentence Transformers](https://www.sbert.net/)
- Vector search by [FAISS](https://github.com/facebookresearch/faiss)
- PDF processing by [PyMuPDF](https://pymupdf.readthedocs.io/)