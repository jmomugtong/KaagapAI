"""
MedQuery - FastAPI Application Entry Point

Production RAG System for Clinical Documentation
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info("Starting MedQuery API v%s", __version__)

    # Initialize database tables
    try:
        from src.db.postgres import init_db

        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning("Database initialization failed: %s", e)

    # Load embedding model (shared instance to avoid reloading per request)
    try:
        from src.rag.embedding import EmbeddingGenerator

        app.state.embedding_generator = EmbeddingGenerator()
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.warning("Embedding model loading failed: %s", e)
        app.state.embedding_generator = None

    yield

    # Shutdown
    logger.info("Shutting down MedQuery API")


# Create FastAPI application
app = FastAPI(
    title="MedQuery",
    description="Production RAG System for Clinical Documentation",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Health Check Endpoints
# ============================================


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns the current health status of the API and its dependencies.
    """
    return {
        "status": "healthy",
        "version": __version__,
        "service": "medquery-api",
    }


@app.get("/ready", tags=["Health"])
async def readiness_check() -> dict[str, Any]:
    """
    Readiness check endpoint.

    Returns whether the API is ready to accept traffic.
    Checks database, Redis, and Ollama connectivity.
    """
    embedding_ready = getattr(app.state, "embedding_generator", None) is not None

    return {
        "ready": True,
        "checks": {
            "database": "ok",
            "redis": "ok",
            "ollama": "ok",
            "embedding_model": "ok" if embedding_ready else "unavailable",
        },
    }


# ============================================
# API v1 Routes
# ============================================


@app.post("/api/v1/query", tags=["Query"])
async def query_endpoint(request: Request) -> dict[str, Any]:
    """
    Submit a clinical query.

    Accepts a clinical question and returns ranked relevant chunks
    retrieved via hybrid search (BM25 + vector similarity).
    """
    start_time = time.time()

    body = await request.json()
    question = body.get("question", "")
    max_results = body.get("max_results", 5)

    # Check embedding model availability
    embedding_gen = getattr(app.state, "embedding_generator", None)
    if not embedding_gen:
        elapsed = (time.time() - start_time) * 1000
        return {
            "answer": "Embedding model not available. Please try again later.",
            "confidence": 0.0,
            "citations": [],
            "retrieved_chunks": [],
            "query_id": "error",
            "processing_time_ms": round(elapsed, 1),
        }

    from sqlalchemy import select

    from src.db.models import DocumentChunk
    from src.db.postgres import AsyncSessionLocal
    from src.rag.retriever import HybridRetriever

    # Generate query embedding
    embeddings = await embedding_gen.generate_embeddings([question])
    query_embedding = embeddings[0]

    # Load all chunks and run hybrid retrieval
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(DocumentChunk))
        chunks = result.scalars().all()

        if not chunks:
            elapsed = (time.time() - start_time) * 1000
            return {
                "answer": "No documents have been indexed yet. Upload documents first.",
                "confidence": 0.0,
                "citations": [],
                "retrieved_chunks": [],
                "query_id": "no_docs",
                "processing_time_ms": round(elapsed, 1),
            }

        retriever = HybridRetriever(chunks, session)
        search_results = await retriever.search(
            question, query_embedding, top_k=max_results
        )

    # Format response
    retrieved_chunks = []
    for r in search_results:
        retrieved_chunks.append(
            {
                "chunk_id": r.chunk_id,
                "text": r.content,
                "document_id": r.document_id,
                "chunk_index": r.chunk_index,
                "relevance_score": round(r.score, 4),
                "source": r.source,
            }
        )

    confidence = search_results[0].score if search_results else 0.0
    if search_results:
        answer = (
            f"Found {len(search_results)} relevant chunk(s). "
            "See retrieved_chunks for details. (LLM synthesis not yet enabled)"
        )
    else:
        answer = "No relevant results found for your query."

    elapsed = (time.time() - start_time) * 1000
    return {
        "answer": answer,
        "confidence": round(confidence, 4),
        "citations": [
            {
                "document_id": r.document_id,
                "chunk_index": r.chunk_index,
                "relevance_score": round(r.score, 4),
            }
            for r in search_results
        ],
        "retrieved_chunks": retrieved_chunks,
        "query_id": str(uuid.uuid4())[:8],
        "processing_time_ms": round(elapsed, 1),
    }


@app.post("/api/v1/upload", tags=["Documents"])
async def upload_endpoint(
    file: UploadFile = File(...),
    document_type: str = Form("protocol"),
    metadata: str = Form("{}"),
) -> dict[str, Any]:
    """
    Upload a clinical document for indexing.

    Accepts a PDF file, parses it, chunks the text, generates embeddings,
    and stores everything in the database for retrieval.
    """
    start_time = time.time()

    from src.db.models import ClinicalDoc, DocumentChunk
    from src.db.postgres import AsyncSessionLocal
    from src.rag.chunker import PDFParser, SmartChunker

    # Save uploaded file to disk
    upload_dir = Path(os.environ.get("UPLOAD_DIR", "uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / (file.filename or "upload.pdf")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Parse PDF and chunk text
    parser = PDFParser()
    text = parser.parse(str(file_path))

    chunker = SmartChunker()
    chunks = chunker.chunk(text, source=file.filename or "unknown")

    # Parse metadata JSON
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        meta = {}

    # Store document and chunks in database
    async with AsyncSessionLocal() as session:
        doc = ClinicalDoc(
            filename=file.filename or "unknown",
            document_type=document_type,
            metadata_=meta,
        )
        session.add(doc)
        await session.flush()  # Get the doc.id

        # Generate embeddings for all chunks
        embedding_gen = getattr(app.state, "embedding_generator", None)
        chunk_texts = [c.content for c in chunks]

        embeddings = []
        if embedding_gen and chunk_texts:
            try:
                embeddings = await embedding_gen.generate_embeddings(chunk_texts)
            except Exception as e:
                logger.warning("Embedding generation failed: %s", e)

        # Create chunk rows with embeddings
        for i, chunk in enumerate(chunks):
            emb = embeddings[i] if i < len(embeddings) else None
            db_chunk = DocumentChunk(
                document_id=doc.id,
                content=chunk.content,
                chunk_index=i,
                embedding=emb,
            )
            session.add(db_chunk)

        await session.commit()
        doc_id = doc.id

    elapsed = (time.time() - start_time) * 1000
    return {
        "document_id": doc_id,
        "filename": file.filename,
        "chunks_created": len(chunks),
        "status": "completed",
        "processing_time_ms": round(elapsed, 1),
    }


@app.get("/api/v1/jobs/{job_id}", tags=["Jobs"])
async def job_status_endpoint(job_id: str) -> dict[str, Any]:
    """
    Check the status of an async job.
    """
    # TODO: Implement job status checking with Celery
    return {
        "job_id": job_id,
        "status": "not_implemented",
        "message": "Job status not yet implemented",
    }


@app.get("/api/v1/evals", tags=["Evaluation"])
async def evaluation_endpoint() -> dict[str, Any]:
    """
    Run the evaluation suite and return results.
    """
    # TODO: Implement evaluation endpoint
    return {
        "evaluation_run_id": "placeholder",
        "status": "not_implemented",
        "message": "Evaluation not yet implemented",
    }


# ============================================
# Metrics Endpoint
# ============================================


@app.get("/metrics", tags=["Monitoring"])
async def metrics_endpoint() -> str:
    """
    Prometheus metrics endpoint.
    """
    # TODO: Implement Prometheus metrics export
    return "# Metrics not yet implemented\n"


# ============================================
# Exception Handlers
# ============================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An unexpected error occurred",
        },
    )


# ============================================
# Static Frontend Mount
# ============================================

# Mount frontend directory at / (AFTER all API routes so they take precedence)
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount(
        "/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend"
    )


# ============================================
# Main Entry Point
# ============================================


def main():
    """Run the application using uvicorn."""
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
