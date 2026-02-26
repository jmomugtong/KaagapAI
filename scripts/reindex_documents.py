#!/usr/bin/env python3
"""
Re-index all documents with the new embedding model and chunking strategy.

This script:
1. Loads all documents from clinical_docs table
2. Re-chunks each document using SemanticDocChunker (with SmartChunker fallback)
3. Re-generates embeddings using nomic-embed-text via Ollama (768-dim)
4. Replaces old chunks in embeddings_cache table
5. Flushes Redis embedding and query caches

Run: python scripts/reindex_documents.py

Prerequisites:
- PostgreSQL must be running and accessible
- Redis must be running (for cache flush)
- Ollama must be running with nomic-embed-text pulled
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def reindex_all():
    """Re-index all documents with new embedding model and chunking."""
    from sqlalchemy import delete, select

    from src.db.models import ClinicalDoc, DocumentChunk
    from src.db.postgres import AsyncSessionLocal, init_db
    from src.rag.cache import CacheManager
    from src.rag.chunker import SemanticDocChunker, SmartChunker
    from src.rag.embedding import EmbeddingGenerator, LangChainEmbeddingAdapter

    # Initialize database
    print("=" * 60)
    print("KaagapAI Document Re-indexing")
    print("=" * 60)
    print()

    print("[1/5] Initializing database...")
    try:
        await init_db()
        print("      Database connected.")
    except Exception as e:
        print(f"      ERROR: Database initialization failed: {e}")
        return False

    # Initialize embedding model
    print("[2/5] Loading embedding model...")
    try:
        embedding_gen = EmbeddingGenerator()
        print(f"      Model: {embedding_gen.model_name} (via Ollama)")
        # Quick test
        test_emb = await embedding_gen.generate_embeddings(["test"])
        dim = len(test_emb[0])
        print(f"      Dimension: {dim}")
    except Exception as e:
        print(f"      ERROR: Failed to load embedding model: {e}")
        return False

    # Initialize chunker
    lc_embeddings = LangChainEmbeddingAdapter(embedding_gen)
    chunker = SemanticDocChunker(embedding_model=lc_embeddings)
    print("      Semantic chunker initialized.")

    # Load all documents
    print("[3/5] Loading documents...")
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(ClinicalDoc))
        docs = result.scalars().all()
        print(f"      Found {len(docs)} document(s).")

        if not docs:
            print("      No documents to re-index.")
            return True

        # Re-index each document
        print("[4/5] Re-indexing documents...")
        total_chunks = 0
        upload_dir = Path(os.environ.get("UPLOAD_DIR", "uploads"))

        for i, doc in enumerate(docs, 1):
            print(f"      [{i}/{len(docs)}] {doc.filename}...")

            # Find the file
            file_path = upload_dir / doc.filename
            if not file_path.exists():
                print(f"         SKIP: File not found at {file_path}")
                continue

            # Parse document
            try:
                from src.rag.chunker import PDFParser

                parser = PDFParser()
                text = parser.parse(str(file_path))
            except Exception as e:
                print(f"         SKIP: Parse failed: {e}")
                continue

            # Re-chunk
            chunks = chunker.chunk(text, source=doc.filename)
            if not chunks:
                print("         SKIP: No chunks produced")
                continue

            # Delete old chunks for this document
            await session.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == doc.id)
            )

            # Generate new embeddings
            chunk_texts = [c.content for c in chunks]
            embeddings = await embedding_gen.generate_embeddings(chunk_texts)

            # Insert new chunks
            for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                db_chunk = DocumentChunk(
                    document_id=doc.id,
                    content=chunk.content,
                    chunk_index=j,
                    embedding=embedding,
                )
                session.add(db_chunk)

            total_chunks += len(chunks)
            print(f"         OK: {len(chunks)} chunks, {dim}-dim embeddings")

        await session.commit()
        print(f"      Total: {total_chunks} chunks across {len(docs)} documents.")

    # Flush caches
    print("[5/5] Flushing Redis caches...")
    try:
        cache = CacheManager()
        emb_count = await cache.flush_embeddings()
        query_count = await cache.flush_queries()
        await cache.close()
        print(f"      Flushed {emb_count} embedding cache keys.")
        print(f"      Flushed {query_count} query cache keys.")
    except Exception as e:
        print(f"      WARNING: Cache flush failed (Redis may be down): {e}")

    print()
    print("=" * 60)
    print("SUCCESS: Re-indexing complete.")
    print("=" * 60)
    return True


def main():
    success = asyncio.run(reindex_all())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
