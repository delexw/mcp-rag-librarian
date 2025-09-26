"""
Document store module for RAG MCP server.

Tracks document changes for incremental knowledge base updates.
"""

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Set, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentStore:
    """SQLite-based store for tracking document changes and metadata."""
    
    def __init__(self, db_path: str = "document_store.db"):
        """
        Initialize document store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_database()
        logger.info(f"Initialized document store: {self.db_path}")
    
    def _init_database(self) -> None:
        """Initialize the SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        filename TEXT NOT NULL,
                        relative_path TEXT PRIMARY KEY,
                        file_hash TEXT NOT NULL,
                        chunk_count INTEGER NOT NULL,
                        last_modified TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                logger.debug("Database tables initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise RuntimeError(f"Database initialization failed: {e}")
    
    def compute_file_hash(self, filepath: Path) -> str:
        """
        Compute SHA-256 hash of a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Hexadecimal hash string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If hash computation fails
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            hash_sha256 = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute hash for {filepath}: {e}")
            raise RuntimeError(f"Hash computation failed: {e}")
    
    def store_document(self, filepath: Path, file_hash: str, chunk_count: int, relative_path: str = None) -> None:
        """
        Store or update document information.

        Args:
            filepath: Path to the document file
            file_hash: SHA-256 hash of the file
            chunk_count: Number of chunks created from the document
            relative_path: Relative path from knowledge base root (optional, defaults to filename)

        Raises:
            RuntimeError: If database operation fails
        """
        try:
            filename = filepath.name
            if relative_path is None:
                relative_path = filename
            last_modified = datetime.fromtimestamp(filepath.stat().st_mtime)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO documents
                    (filename, relative_path, file_hash, chunk_count, last_modified)
                    VALUES (?, ?, ?, ?, ?)
                """, (filename, relative_path, file_hash, chunk_count, last_modified))
                conn.commit()

            logger.debug(f"Stored document info: {relative_path}")
            
        except Exception as e:
            logger.error(f"Failed to store document {relative_path or filepath.name}: {e}")
            raise RuntimeError(f"Failed to store document: {e}")
    
    def is_document_changed(self, filepath: Path, relative_path: str = None) -> Tuple[bool, str]:
        """
        Check if a document has changed since last indexing.

        Args:
            filepath: Path to the document file
            relative_path: Relative path from knowledge base root (optional, defaults to filename)

        Returns:
            Tuple of (has_changed: bool, current_hash: str)

        Raises:
            RuntimeError: If check operation fails
        """
        try:
            if relative_path is None:
                relative_path = filepath.name
            current_hash = self.compute_file_hash(filepath)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT file_hash FROM documents WHERE relative_path = ?",
                    (relative_path,)
                )
                result = cursor.fetchone()

            if result is None:
                # Document is new
                return True, current_hash

            stored_hash = result[0]
            has_changed = stored_hash != current_hash

            return has_changed, current_hash

        except Exception as e:
            logger.error(f"Failed to check document {relative_path}: {e}")
            raise RuntimeError(f"Failed to check document: {e}")
    
    def remove_document(self, relative_path: str) -> bool:
        """
        Remove document from store.

        Args:
            relative_path: Relative path of the document file

        Returns:
            True if document was removed, False if it didn't exist

        Raises:
            RuntimeError: If removal operation fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM documents WHERE relative_path = ?",
                    (relative_path,)
                )
                removed = cursor.rowcount > 0
                conn.commit()

            if removed:
                logger.debug(f"Removed document: {relative_path}")

            return removed

        except Exception as e:
            logger.error(f"Failed to remove document {relative_path}: {e}")
            raise RuntimeError(f"Failed to remove document: {e}")
    
    def get_all_document_names(self) -> Set[str]:
        """
        Get set of all document relative paths in the store.

        Returns:
            Set of document relative paths

        Raises:
            RuntimeError: If query operation fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT relative_path FROM documents")
                relative_paths = {row[0] for row in cursor.fetchall()}

            return relative_paths

        except Exception as e:
            logger.error(f"Failed to get document names: {e}")
            raise RuntimeError(f"Failed to get document names: {e}")
    
    def get_document_info(self, relative_path: str) -> Optional[dict]:
        """
        Get information about a specific document.

        Args:
            relative_path: Relative path of the document file

        Returns:
            Dictionary with document info or None if not found

        Raises:
            RuntimeError: If query operation fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT filename, relative_path, file_hash, chunk_count, last_modified, created_at
                    FROM documents WHERE relative_path = ?
                """, (relative_path,))
                result = cursor.fetchone()

            if result is None:
                return None

            return {
                "filename": result[0],
                "relative_path": result[1],
                "file_hash": result[2],
                "chunk_count": result[3],
                "last_modified": result[4],
                "created_at": result[5]
            }
            
        except Exception as e:
            logger.error(f"Failed to get document info for {relative_path}: {e}")
            raise RuntimeError(f"Failed to get document info: {e}")
    
    def get_stats(self) -> dict:
        """
        Get statistics about the document store.
        
        Returns:
            Dictionary with store statistics
            
        Raises:
            RuntimeError: If query operation fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_documents,
                        SUM(chunk_count) as total_chunks,
                        MAX(last_modified) as latest_modification
                    FROM documents
                """)
                result = cursor.fetchone()
            
            return {
                "total_documents": result[0] or 0,
                "total_chunks": result[1] or 0,
                "latest_modification": result[2],
                "database_path": str(self.db_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get store statistics: {e}")
            raise RuntimeError(f"Failed to get store statistics: {e}")
    
    def clear(self) -> None:
        """
        Clear all documents from the store.
        
        Raises:
            RuntimeError: If clear operation fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM documents")
                conn.commit()
            
            logger.info("Cleared document store")
            
        except Exception as e:
            logger.error(f"Failed to clear document store: {e}")
            raise RuntimeError(f"Failed to clear document store: {e}")
