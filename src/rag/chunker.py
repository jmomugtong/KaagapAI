"""
MedQuery Document Chunker Module

PDF parsing and smart text chunking for the RAG pipeline.
Supports PyPDF2 with pdfplumber fallback for complex layouts.
"""

import hashlib
import io
import re
from pathlib import Path
from typing import Any

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader

# ============================================
# Exceptions
# ============================================


class PDFParseError(Exception):
    """Raised when PDF parsing fails."""

    pass


# ============================================
# Data Models
# ============================================


class DocumentChunk(BaseModel):
    """Represents a chunk of document text with metadata.

    Attributes:
        text: The actual text content of the chunk.
        chunk_index: Zero-based index of this chunk in the document.
        chunk_hash: SHA256 hash of the text content for deduplication.
        metadata: Additional metadata (source, section_title, etc.).
    """

    text: str = Field(..., description="The text content of the chunk")
    chunk_index: int = Field(..., ge=0, description="Zero-based chunk index")
    chunk_hash: str = Field(..., description="SHA256 hash of text content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Chunk metadata"
    )


# ============================================
# PDF Parser
# ============================================


class PDFParser:
    """Parses PDF documents to extract text content.

    Uses PyPDF2 as the primary parser with pdfplumber as fallback
    for complex layouts (tables, scanned documents, etc.).
    """

    def __init__(self) -> None:
        """Initialize the PDF parser."""
        pass

    def parse(self, pdf_bytes: bytes) -> str:
        """Parse PDF from bytes and extract text.

        Args:
            pdf_bytes: Raw PDF file content as bytes.

        Returns:
            Extracted text content from the PDF.

        Raises:
            PDFParseError: If PDF cannot be parsed by any method.
        """
        if not pdf_bytes:
            raise PDFParseError("Empty PDF data provided")

        # Try PyPDF2 first
        try:
            text = self._extract_with_pypdf2(pdf_bytes)
            if text.strip():
                return text
        except Exception:
            pass

        # Fallback to pdfplumber
        try:
            text = self._extract_with_pdfplumber(pdf_bytes)
            if text.strip():
                return text
        except Exception:
            pass

        # If both fail and we have valid PDF header, return empty
        if pdf_bytes.startswith(b"%PDF"):
            return ""

        raise PDFParseError("Failed to parse PDF with any available method")

    def parse_file(self, file_path: str | Path) -> str:
        """Parse PDF from file path.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text content from the PDF.

        Raises:
            PDFParseError: If file cannot be read or parsed.
        """
        path = Path(file_path)
        if not path.exists():
            raise PDFParseError(f"PDF file not found: {file_path}")

        try:
            pdf_bytes = path.read_bytes()
            return self.parse(pdf_bytes)
        except PDFParseError:
            raise
        except Exception as e:
            raise PDFParseError(f"Failed to read PDF file: {e}") from e

    def _extract_with_pypdf2(self, pdf_bytes: bytes) -> str:
        """Extract text using PyPDF2.

        Args:
            pdf_bytes: Raw PDF content.

        Returns:
            Extracted text from all pages.
        """
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        return "\n\n".join(text_parts)

    def _extract_with_pdfplumber(self, pdf_bytes: bytes) -> str:
        """Extract text using pdfplumber (better for tables).

        Args:
            pdf_bytes: Raw PDF content.

        Returns:
            Extracted text from all pages.
        """
        text_parts = []

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        return "\n\n".join(text_parts)


# ============================================
# Metadata Extractor
# ============================================


class MetadataExtractor:
    """Extracts metadata from document chunks.

    Identifies section titles from markdown headers and numbered sections.
    """

    # Regex patterns for section headers
    MARKDOWN_HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    NUMBERED_SECTION_PATTERN = re.compile(
        r"^(\d+(?:\.\d+)*)\s+(.+)$", re.MULTILINE
    )

    def __init__(self) -> None:
        """Initialize the metadata extractor."""
        pass

    def extract(
        self,
        text: str,
        source: str,
        chunk_index: int,
        additional_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract metadata from a text chunk.

        Args:
            text: The chunk text content.
            source: Source document filename.
            chunk_index: Index of this chunk in the document.
            additional_metadata: Optional extra metadata to include.

        Returns:
            Dictionary containing extracted metadata.
        """
        metadata: dict[str, Any] = {
            "source": source,
            "chunk_index": chunk_index,
            "section_title": self._extract_section_title(text),
        }

        if additional_metadata:
            metadata.update(additional_metadata)

        return metadata

    def _extract_section_title(self, text: str) -> str:
        """Extract section title from text.

        Args:
            text: Text to extract title from.

        Returns:
            Section title if found, otherwise "Unknown".
        """
        # Try markdown headers first
        match = self.MARKDOWN_HEADER_PATTERN.search(text)
        if match:
            return match.group(2).strip()

        # Try numbered sections
        match = self.NUMBERED_SECTION_PATTERN.search(text)
        if match:
            return match.group(2).strip()

        # Check for first line as potential title
        first_line = text.strip().split("\n")[0].strip()
        if first_line and len(first_line) < 100:
            return first_line

        return "Unknown"


# ============================================
# Smart Chunker
# ============================================


class SmartChunker:
    """Smart text chunker with section-aware splitting.

    Splits text at section boundaries when possible, maintaining
    context overlap between chunks for better retrieval.

    Attributes:
        max_chunk_size: Maximum tokens per chunk (default: 512).
        overlap_size: Token overlap between chunks (default: 100).
    """

    # Section header patterns for splitting
    SECTION_SEPARATORS = [
        "\n# ",      # H1
        "\n## ",     # H2
        "\n### ",    # H3
        "\n#### ",   # H4
        "\n\n",      # Double newline (paragraph)
        "\n",        # Single newline
        ". ",        # Sentence boundary
        " ",         # Word boundary
    ]

    def __init__(
        self,
        max_chunk_size: int = 512,
        overlap_size: int = 100,
    ) -> None:
        """Initialize the smart chunker.

        Args:
            max_chunk_size: Maximum tokens (roughly words) per chunk.
            overlap_size: Number of overlapping tokens between chunks.
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

        # Use LangChain's splitter for the heavy lifting
        # Approximate 4 chars per token
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size * 4,
            chunk_overlap=overlap_size * 4,
            separators=self.SECTION_SEPARATORS,
            keep_separator=True,
            length_function=len,
        )

        self._metadata_extractor = MetadataExtractor()

    def chunk(
        self,
        text: str,
        source: str,
        additional_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Chunk text into document chunks with metadata.

        Args:
            text: The full document text to chunk.
            source: Source document filename.
            additional_metadata: Optional metadata to include in all chunks.

        Returns:
            List of DocumentChunk objects.
        """
        if not text or not text.strip():
            return []

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split into chunks
        raw_chunks = self._splitter.split_text(text)

        # Create DocumentChunk objects
        chunks: list[DocumentChunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue

            chunk_hash = self._compute_hash(chunk_text)
            metadata = self._metadata_extractor.extract(
                text=chunk_text,
                source=source,
                chunk_index=idx,
                additional_metadata=additional_metadata,
            )

            chunk = DocumentChunk(
                text=chunk_text,
                chunk_index=idx,
                chunk_hash=chunk_hash,
                metadata=metadata,
            )
            chunks.append(chunk)

        # Re-index to ensure sequential indices
        for idx, chunk in enumerate(chunks):
            chunk.chunk_index = idx
            chunk.metadata["chunk_index"] = idx

        return chunks

    def _compute_hash(self, text: str) -> str:
        """Compute SHA256 hash of text content.

        Args:
            text: Text to hash.

        Returns:
            Hexadecimal hash string.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================
# Convenience Functions
# ============================================


def parse_and_chunk_pdf(
    pdf_source: bytes | str | Path,
    source_name: str | None = None,
    max_chunk_size: int = 512,
    overlap_size: int = 100,
) -> list[DocumentChunk]:
    """Convenience function to parse PDF and chunk in one step.

    Args:
        pdf_source: PDF bytes, file path string, or Path object.
        source_name: Document name for metadata. If None, uses filename.
        max_chunk_size: Maximum tokens per chunk.
        overlap_size: Token overlap between chunks.

    Returns:
        List of DocumentChunk objects.

    Raises:
        PDFParseError: If PDF cannot be parsed.
    """
    parser = PDFParser()
    chunker = SmartChunker(max_chunk_size=max_chunk_size, overlap_size=overlap_size)

    # Parse PDF
    if isinstance(pdf_source, bytes):
        text = parser.parse(pdf_source)
        source = source_name or "document.pdf"
    else:
        path = Path(pdf_source)
        text = parser.parse_file(path)
        source = source_name or path.name

    # Chunk text
    return chunker.chunk(text, source=source)
