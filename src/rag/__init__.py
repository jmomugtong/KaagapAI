"""
MedQuery RAG Module

Document ingestion pipeline for clinical documentation search.
Provides PDF parsing, smart chunking, embedding generation, and caching.
"""

from src.rag.cache import CacheManager
from src.rag.chunker import (
    DocumentChunk,
    MetadataExtractor,
    PDFParseError,
    PDFParser,
    SmartChunker,
    parse_and_chunk_pdf,
)
from src.rag.embedding import (
    EMBEDDING_DIMENSION,
    EmbeddingGenerator,
    embed_text,
    embed_texts,
)

__all__ = [
    # Chunker
    "PDFParser",
    "PDFParseError",
    "SmartChunker",
    "MetadataExtractor",
    "DocumentChunk",
    "parse_and_chunk_pdf",
    # Embedding
    "EmbeddingGenerator",
    "EMBEDDING_DIMENSION",
    "embed_text",
    "embed_texts",
    # Cache
    "CacheManager",
]
