"""
Document processing module for RAG MCP server.

Handles document loading, text extraction, and chunking operations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

logger = logging.getLogger(__name__)


class Document:
    """Represents a document chunk with metadata."""

    def __init__(
        self, filename: str, chunk_index: int, content: str, relative_path: Optional[str] = None
    ):
        """
        Initialize a document chunk.

        Args:
            filename: Source filename
            chunk_index: Index of this chunk within the source file
            content: Text content of the chunk
            relative_path: Relative path from knowledge base root (optional)
        """
        self.filename = filename
        self.chunk_index = chunk_index
        self.content = content
        self.relative_path = relative_path or filename

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create a Document from a dictionary."""
        return cls(
            filename=data["filename"],
            chunk_index=data["chunk_index"],
            content=data["content"],
            relative_path=data.get("relative_path"),
        )

    def __str__(self) -> str:
        return f"Document(filename={self.filename}, relative_path={self.relative_path}, chunk_index={self.chunk_index}, content_length={len(self.content)})"


class DocumentProcessor:
    """Handles document loading, chunking, and processing."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(
        self, directory: Union[str, Path], extensions: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load all documents with specified extensions from a directory.

        Args:
            directory: Path to knowledge base directory
            extensions: List of file extensions to process (default: [".txt", ".pdf", ".md"])

        Returns:
            List of Document objects

        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If no valid files found
        """
        if extensions is None:
            extensions = self.get_supported_extensions()

        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory '{directory}' does not exist")

        documents = []
        logger.info(f"Loading documents from {directory}...")

        # Use the centralized file discovery method
        files = self.find_documents(directory, extensions)

        if not files:
            raise ValueError(f"No files with extensions {extensions} found in {directory}")

        for filepath in tqdm(files, desc="Processing files"):
            try:
                # Calculate relative path from the knowledge base directory
                relative_path = filepath.relative_to(directory)
                file_documents = self.process_file(filepath, str(relative_path))
                documents.extend(file_documents)
                logger.debug(f"Processed {relative_path}: {len(file_documents)} chunks")
            except Exception as e:
                logger.error(
                    f"Error processing {filepath.relative_to(directory) if filepath.is_relative_to(directory) else filepath.name}: {e}"
                )

        logger.info(f"Loaded {len(documents)} document chunks from {len(files)} files")
        return documents

    def process_file(self, filepath: Path, relative_path: Optional[str] = None) -> List[Document]:
        """
        Process a single file and create document chunks.

        Args:
            filepath: Path to the file to process
            relative_path: Relative path from knowledge base root

        Returns:
            List of Document chunks from the file

        Raises:
            ValueError: If file type is not supported
        """
        documents = []
        if relative_path is None:
            relative_path = filepath.name

        try:
            content = self._extract_content(filepath)
            if content:
                chunks = self._chunk_text(content)
                for idx, chunk in enumerate(chunks):
                    documents.append(
                        Document(
                            filename=filepath.name,
                            chunk_index=idx,
                            content=chunk,
                            relative_path=relative_path,
                        )
                    )
                logger.debug(f"Processed {relative_path}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error processing {relative_path}: {e}")
            raise

        return documents

    def _extract_content(self, filepath: Path) -> str:
        """
        Extract text content from a file based on its extension.

        Args:
            filepath: Path to the file

        Returns:
            Extracted text content

        Raises:
            ValueError: If file type is not supported
        """
        ext = filepath.suffix.lower()

        if ext in [".txt", ".md"]:
            return self._extract_text_from_txt(filepath)
        elif ext == ".pdf":
            return self._extract_text_from_pdf(filepath)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _extract_text_from_txt(self, filepath: Path) -> str:
        """Extract text from a .txt file."""
        try:
            return filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            try:
                return filepath.read_text(encoding="latin-1")
            except Exception as e:
                logger.error(f"Failed to read {filepath.name} with multiple encodings: {e}")
                raise
        except Exception as e:
            logger.error(f"Error reading {filepath.name}: {e}")
            raise

    def _extract_text_from_pdf(self, filepath: Path) -> str:
        """Extract text from a PDF file using PyMuPDF."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(filepath)
            text = ""

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()

            doc.close()
            return text

        except Exception as e:
            logger.error(f"Error extracting PDF {filepath.name}: {e}")
            raise

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to split into chunks

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Avoid splitting in the middle of words if possible
            if end < text_len:
                # Look for the last space within the last 100 characters of the chunk
                search_start = max(end - 100, start)
                last_100_chars = text[search_start:end]
                last_space = last_100_chars.rfind(" ")

                if last_space != -1:
                    # Adjust end to the position of that space in the original text
                    end = search_start + last_space

            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def find_documents(
        self, directory: Union[str, Path], extensions: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Recursively find all supported document files in a directory.

        Args:
            directory: Path to directory to search
            extensions: List of file extensions to search for (optional, defaults to supported extensions)

        Returns:
            List of Path objects for all matching files

        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        if extensions is None:
            extensions = self.get_supported_extensions()

        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory '{directory}' does not exist")

        files = []
        for ext in extensions:
            pattern = f"**/*{ext}"
            matched_files = list(directory.glob(pattern))
            files.extend([f for f in matched_files if f.is_file()])

        return files

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return [".txt", ".pdf", ".md"]
