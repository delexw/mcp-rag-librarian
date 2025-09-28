"""
Pydantic response models for MCP tools.

These models define the structured return types for all RAG MCP server tools,
replacing manual JSON string creation with type-safe Pydantic models.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response model for all tool responses."""

    success: bool = Field(description="Whether the operation was successful")
    operation: str = Field(description="The operation that was performed")
    message: Optional[str] = Field(default=None, description="Optional message about the operation")


class ErrorResponse(BaseResponse):
    """Error response model."""

    success: bool = Field(default=False, description="Always False for error responses")
    error: str = Field(description="Error code")
    path: Optional[str] = Field(default=None, description="Path related to the error")


class DocumentInfo(BaseModel):
    """Document information model."""

    files: int = Field(description="Number of unique files")
    chunks: int = Field(description="Number of document chunks")


class ConfigurationInfo(BaseModel):
    """Configuration information model."""

    chunk_size: int = Field(description="Size of text chunks")
    chunk_overlap: int = Field(description="Overlap between chunks")
    embedding_model: str = Field(description="Name of the embedding model used")
    embedding_dimension: int = Field(description="Dimension of embeddings")


class InitializeKnowledgeBaseResponse(BaseResponse):
    """Response model for initialize_knowledge_base tool."""

    success: bool = Field(default=True)
    operation: str = Field(default="initialize_knowledge_base")
    path: str = Field(description="Path to the knowledge base")
    documents: DocumentInfo = Field(description="Document processing statistics")
    configuration: ConfigurationInfo = Field(description="Knowledge base configuration")
    warnings: Optional[List[str]] = Field(default=None, description="Optional warnings")


class SearchResult(BaseModel):
    """Individual search result model."""

    rank: int = Field(description="Rank of the result (1-based)")
    filename: str = Field(description="Name of the source file")
    relative_path: str = Field(description="Relative path to the source file")
    content: str = Field(description="Content of the matching chunk")


# SemanticSearchResponse is now just List[SearchResult] - defined in tool_registry.py


class IndexInfo(BaseModel):
    """FAISS index information model."""

    status: str = Field(description="Index status (initialized/not_initialized)")
    dimension: int = Field(description="Vector dimension")
    total_vectors: int = Field(description="Total number of vectors in the index")
    index_type: str = Field(description="Type of FAISS index")


class EmbeddingModelInfo(BaseModel):
    """Embedding model information model."""

    name: str = Field(description="Name of the embedding model")
    dimension: int = Field(description="Embedding dimension")
    max_sequence_length: Optional[int] = Field(default=None, description="Maximum sequence length")


class SearchConfigInfo(BaseModel):
    """Search configuration information model."""

    default_top_k: int = Field(description="Default number of search results")


class DocumentStoreInfo(BaseModel):
    """Document store information model."""

    database_path: str = Field(description="Path to the document store database")
    tracked_documents: int = Field(description="Number of tracked documents")
    tracked_chunks: int = Field(description="Number of tracked chunks")
    last_modification: Optional[str] = Field(
        default=None, description="Last modification timestamp"
    )


class ChunkingConfigInfo(BaseModel):
    """Chunking configuration information model."""

    chunk_size: int = Field(description="Size of text chunks")
    chunk_overlap: int = Field(description="Overlap between chunks")


class DocumentStats(BaseModel):
    """Document statistics model."""

    total_files: int = Field(description="Total number of files")
    total_chunks: int = Field(description="Total number of chunks")
    average_chunks_per_file: float = Field(description="Average chunks per file")


class KnowledgeBaseStatsResponse(BaseResponse):
    """Response model for get_knowledge_base_stats tool."""

    success: bool = Field(default=True)
    operation: str = Field(default="get_knowledge_base_stats")
    path: str = Field(description="Path to the knowledge base")
    documents: DocumentStats = Field(description="Document statistics")
    chunking_configuration: ChunkingConfigInfo = Field(description="Chunking configuration")
    index: IndexInfo = Field(description="FAISS index information")
    embedding_model: EmbeddingModelInfo = Field(description="Embedding model information")
    search_configuration: SearchConfigInfo = Field(description="Search configuration")
    document_store: DocumentStoreInfo = Field(description="Document store information")


class DocumentListItem(BaseModel):
    """Individual document in the list."""

    relative_path: str = Field(description="Relative path to the document")
    chunk_count: int = Field(description="Number of chunks for this document")
    total_content_length: int = Field(description="Total content length")
    last_modified: Optional[str] = Field(default=None, description="Last modification timestamp")
    file_hash: Optional[str] = Field(default=None, description="File hash for change detection")


class ListDocumentsResponse(BaseResponse):
    """Response model for list_documents tool."""

    success: bool = Field(default=True)
    operation: str = Field(default="list_documents")
    path: str = Field(description="Path to the knowledge base")
    total_documents: int = Field(description="Total number of documents")
    documents: List[DocumentListItem] = Field(description="List of documents")


class RefreshKnowledgeBaseResponse(BaseResponse):
    """Response model for refresh_knowledge_base tool."""

    success: bool = Field(default=True)
    operation: str = Field(default="refresh_knowledge_base")
    path: str = Field(description="Path to the knowledge base")
    hasChanges: bool = Field(description="Whether changes were detected and processed")
    documents: DocumentInfo = Field(description="Document processing statistics")
    configuration: ConfigurationInfo = Field(description="Knowledge base configuration")
    status: str = Field(description="Status description of the refresh operation")


# Union type for all possible responses
ToolResponse = Union[
    InitializeKnowledgeBaseResponse,
    List[SearchResult],  # SemanticSearchResponse is now just List[SearchResult]
    KnowledgeBaseStatsResponse,
    ListDocumentsResponse,
    RefreshKnowledgeBaseResponse,
    ErrorResponse,
]
