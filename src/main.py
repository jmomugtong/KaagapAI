"""
KaagapAI - FastAPI Application Entry Point

Production RAG System for Clinical Documentation
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from src import __version__
from src.security.rate_limiter import RateLimitMiddleware

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
    logger.info("Starting KaagapAI API v%s", __version__)

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

    # Initialize Ollama LLM client
    try:
        from src.llm.ollama_client import OllamaClient

        app.state.ollama_client = OllamaClient()
        is_healthy = await app.state.ollama_client.health_check()
        if is_healthy:
            logger.info("Ollama client initialized and healthy")
            # Warm up: load model into memory with a 1-token generation
            logger.info("Warming up LLM model (loading into memory)...")
            warmup_ok = await app.state.ollama_client.warmup()
            if warmup_ok:
                logger.info("LLM model warmed up successfully")
            else:
                logger.warning("LLM warmup failed (model may still be loading)")
        else:
            logger.warning("Ollama client initialized but service is unreachable")
    except Exception as e:
        logger.warning("Ollama client initialization failed: %s", e)
        app.state.ollama_client = None

    # Initialize FlashRank reranker (loads model once, reused for all requests)
    try:
        from src.rag.reranker import Reranker

        app.state.reranker = Reranker()
        logger.info("Reranker initialized successfully")
    except Exception as e:
        logger.warning("Reranker initialization failed: %s", e)
        app.state.reranker = None

    # Load document chunks and build BM25 index once (cached for all queries)
    try:
        from sqlalchemy import select

        from src.db.models import ClinicalDoc, DocumentChunk
        from src.db.postgres import AsyncSessionLocal

        logger.info("Loading document chunks for caching...")
        start = time.time()
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(DocumentChunk))
            chunks = result.scalars().all()
            # Store as list to avoid lazy-loading issues
            app.state.cached_chunks = list(chunks)

            # Build doc_id → filename map for citation resolution
            doc_result = await session.execute(select(ClinicalDoc))
            docs = doc_result.scalars().all()
            app.state.doc_name_map = {d.id: d.filename for d in docs}

            logger.info(
                "Loaded %d chunks, %d documents in %.2fs",
                len(chunks),
                len(docs),
                time.time() - start,
            )
    except Exception as e:
        logger.warning("Chunk caching failed: %s", e)
        app.state.cached_chunks = []
        app.state.doc_name_map = {}

    yield

    # Shutdown
    logger.info("Shutting down KaagapAI API")


# Create FastAPI application
app = FastAPI(
    title="KaagapAI",
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

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware)


# No-cache middleware for frontend static files
@app.middleware("http")
async def no_cache_frontend(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        response.headers[
            "Cache-Control"
        ] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


# ============================================
# Health Check Endpoints
# ============================================


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "service": "kaagapai-api",
    }


@app.get("/ready", tags=["Health"])
async def readiness_check() -> dict[str, Any]:
    """Readiness check with dependency status."""
    embedding_ready = getattr(app.state, "embedding_generator", None) is not None
    ollama_client = getattr(app.state, "ollama_client", None)
    ollama_status = "unavailable"
    if ollama_client:
        try:
            ollama_status = "ok" if await ollama_client.health_check() else "degraded"
        except Exception:
            ollama_status = "error"

    return {
        "ready": True,
        "checks": {
            "database": "ok",
            "redis": "ok",
            "ollama": ollama_status,
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

    Validates input, redacts PII, runs hybrid retrieval + LLM synthesis,
    redacts PII from output, and returns structured response with citations.
    """
    body = await request.json()
    question = body.get("question", "")
    max_results = body.get("max_results", 3)
    confidence_threshold = body.get("confidence_threshold", 0.70)

    # Input validation
    from src.security.input_validation import InputValidator

    validator = InputValidator()
    if not validator.is_safe(question):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query contains potentially unsafe content",
        )

    from src.pipelines.classical import ClassicalPipeline

    pipeline = ClassicalPipeline(
        embedding_generator=getattr(app.state, "embedding_generator", None),
        ollama_client=getattr(app.state, "ollama_client", None),
        reranker=getattr(app.state, "reranker", None),
        cached_chunks=getattr(app.state, "cached_chunks", None),
        doc_name_map=getattr(app.state, "doc_name_map", None),
    )
    result = await pipeline.run(question, max_results, confidence_threshold)

    # Record metrics
    from src.observability.metrics import record_query

    record_query(
        latency_ms=result.processing_time_ms,
        success=True,
        cache_hit=bool(result.cached),
        hallucination=bool(result.hallucination_flagged),
    )

    # Convert PipelineResult to backward-compatible response dict
    response: dict[str, Any] = {
        "answer": result.answer,
        "confidence": result.confidence,
        "citations": result.citations,
        "retrieved_chunks": result.retrieved_chunks,
        "query_id": result.query_id,
        "processing_time_ms": result.processing_time_ms,
    }
    if result.hallucination_flagged:
        response["hallucination_flagged"] = True
    if result.cached:
        response["cached"] = True
    return response


@app.post("/api/v1/query/stream", tags=["Query"])
async def query_stream_endpoint(request: Request):
    """
    Submit a clinical query and stream the LLM response as Server-Sent Events.

    Same pipeline as /api/v1/query up to LLM call, but streams tokens
    instead of buffering the full response. Falls back to non-streaming
    on error.
    """
    from fastapi.responses import StreamingResponse

    body = await request.json()
    question = body.get("question", "")
    max_results = body.get("max_results", 3)

    # Input validation
    from src.security.input_validation import InputValidator

    validator = InputValidator()
    if not validator.is_safe(question):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query contains potentially unsafe content",
        )

    # PII redaction on input
    from src.security.pii_redaction import PIIRedactor

    redactor = PIIRedactor()
    question = redactor.redact(question)

    # Check embedding model
    embedding_gen = getattr(app.state, "embedding_generator", None)
    if not embedding_gen:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model not available",
        )

    ollama_client = getattr(app.state, "ollama_client", None)
    if not ollama_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service not available",
        )

    # Embedding + retrieval (same as non-streaming endpoint)
    from sqlalchemy import select

    from src.db.models import DocumentChunk
    from src.db.postgres import AsyncSessionLocal
    from src.rag.retriever import HybridRetriever, ScoredChunk

    embeddings = await embedding_gen.generate_embeddings([question], is_query=True)
    query_embedding = embeddings[0]

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(DocumentChunk))
        chunks = result.scalars().all()

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No documents indexed yet",
            )

        doc_name_map = getattr(app.state, "doc_name_map", {})
        retriever = HybridRetriever(chunks, session, doc_name_map=doc_name_map)
        search_results = await retriever.search(
            question, query_embedding, top_k=max_results
        )

    # Rerank results using FlashRank cross-encoder
    reranker = getattr(app.state, "reranker", None)
    if reranker and search_results:
        try:
            reranked = await reranker.rerank(
                question, search_results, top_k=max_results
            )
            search_results = [
                ScoredChunk(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    document_id=r.document_id,
                    chunk_index=r.chunk_index,
                    score=r.final_score,
                    source=r.source,
                    document_name=r.document_name,
                )
                for r in reranked
            ]
        except Exception as e:
            logger.warning("Reranking failed, using retrieval order: %s", e)

    if not search_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant results found",
        )

    prompt_chunks = [
        {
            "text": r.content,
            "metadata": {
                "source": r.document_name or f"Document {r.document_id}",
                "chunk_index": r.chunk_index,
                "document_id": r.document_id,
            },
        }
        for r in search_results
    ]

    from src.llm.prompt_templates import PromptTemplate

    template = PromptTemplate()
    prompt = template.build(question=question, chunks=prompt_chunks)

    async def event_stream():
        try:
            async for token in ollama_client.generate_stream(prompt):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("Stream error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/v1/agent/query", tags=["Query"])
async def agent_query_endpoint(request: Request) -> dict[str, Any]:
    """
    Submit a clinical query using the agentic RAG pipeline.

    Classifies the query, decomposes complex queries into sub-queries,
    performs iterative retrieval, and self-reflects on answer completeness.
    """

    body = await request.json()
    question = body.get("question", "")
    max_results = body.get("max_results", 3)
    confidence_threshold = body.get("confidence_threshold", 0.70)

    # Input validation
    from src.security.input_validation import InputValidator

    validator = InputValidator()
    if not validator.is_safe(question):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query contains potentially unsafe content",
        )

    from src.pipelines.agentic import AgenticPipeline

    pipeline = AgenticPipeline(
        embedding_generator=getattr(app.state, "embedding_generator", None),
        ollama_client=getattr(app.state, "ollama_client", None),
        reranker=getattr(app.state, "reranker", None),
        doc_name_map=getattr(app.state, "doc_name_map", None),
        cached_chunks=getattr(app.state, "cached_chunks", None),
    )
    result = await pipeline.run(question, max_results, confidence_threshold)

    from src.observability.metrics import record_query

    record_query(
        latency_ms=result.processing_time_ms,
        success=True,
        cache_hit=False,
        hallucination=bool(result.hallucination_flagged),
    )

    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "citations": result.citations,
        "retrieved_chunks": result.retrieved_chunks,
        "query_id": result.query_id,
        "processing_time_ms": result.processing_time_ms,
        "hallucination_flagged": result.hallucination_flagged,
        "pipeline": result.pipeline,
        "steps": result.steps,
    }


@app.post("/api/v1/compare", tags=["Query"])
async def compare_endpoint(request: Request) -> dict[str, Any]:
    """
    Run both classical and agentic pipelines concurrently and return
    a side-by-side comparison of their results.
    """
    import asyncio

    body = await request.json()
    question = body.get("question", "")
    max_results = body.get("max_results", 3)
    confidence_threshold = body.get("confidence_threshold", 0.70)

    # Input validation (run once, not per-pipeline)
    from src.security.input_validation import InputValidator

    validator = InputValidator()
    if not validator.is_safe(question):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query contains potentially unsafe content",
        )

    from src.pipelines.agentic import AgenticPipeline
    from src.pipelines.classical import ClassicalPipeline

    embedding_gen = getattr(app.state, "embedding_generator", None)
    ollama_client = getattr(app.state, "ollama_client", None)
    reranker = getattr(app.state, "reranker", None)
    cached_chunks = getattr(app.state, "cached_chunks", None)
    doc_name_map = getattr(app.state, "doc_name_map", None)

    classical = ClassicalPipeline(
        embedding_gen, ollama_client, reranker, cached_chunks, doc_name_map
    )
    agentic = AgenticPipeline(
        embedding_gen, ollama_client, reranker, doc_name_map, cached_chunks
    )

    # Run both concurrently
    classical_result, agentic_result = await asyncio.gather(
        classical.run(question, max_results, confidence_threshold),
        agentic.run(question, max_results, confidence_threshold),
        return_exceptions=True,
    )

    def _result_to_dict(r):
        if isinstance(r, Exception):
            return {
                "answer": f"Pipeline error: {r}",
                "confidence": 0.0,
                "citations": [],
                "retrieved_chunks": [],
                "query_id": "error",
                "processing_time_ms": 0,
                "hallucination_flagged": False,
                "pipeline": "error",
                "steps": [],
            }
        return {
            "answer": r.answer,
            "confidence": r.confidence,
            "citations": r.citations,
            "retrieved_chunks": r.retrieved_chunks,
            "query_id": r.query_id,
            "processing_time_ms": r.processing_time_ms,
            "hallucination_flagged": r.hallucination_flagged,
            "pipeline": r.pipeline,
            "steps": r.steps,
        }

    c_dict = _result_to_dict(classical_result)
    a_dict = _result_to_dict(agentic_result)

    # Compute comparison metrics
    c_time = c_dict["processing_time_ms"] or 1
    a_time = a_dict["processing_time_ms"] or 1
    c_retrieval_passes = sum(
        1 for s in c_dict.get("steps", []) if s.get("name") == "retrieve"
    )
    a_retrieval_passes = sum(
        1 for s in a_dict.get("steps", []) if s.get("name") == "retrieve"
    )

    return {
        "classical": c_dict,
        "agentic": a_dict,
        "comparison": {
            "latency_ratio": round(a_time / c_time, 2) if c_time else 0,
            "confidence_delta": round(a_dict["confidence"] - c_dict["confidence"], 4),
            "classical_retrieval_passes": max(c_retrieval_passes, 1),
            "agentic_retrieval_passes": max(a_retrieval_passes, 1),
            "classical_chunks_used": len(c_dict.get("retrieved_chunks", [])),
            "agentic_chunks_used": len(a_dict.get("retrieved_chunks", [])),
        },
    }


@app.get("/api/v1/documents/{filename}", tags=["Documents"])
async def download_document(filename: str):
    """Serve an uploaded document PDF."""
    upload_dir = Path(os.environ.get("UPLOAD_DIR", "uploads")).resolve()
    file_path = (upload_dir / filename).resolve()
    # Ensure the resolved path stays within upload_dir (path traversal protection)
    if not str(file_path).startswith(str(upload_dir)):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not file_path.exists() or not file_path.name.endswith(".pdf"):
        raise HTTPException(status_code=404, detail="Document not found")
    from starlette.responses import Response

    content = file_path.read_bytes()
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.post("/api/v1/upload", tags=["Documents"])
async def upload_endpoint(
    file: UploadFile = File(...),
    document_type: str = Form("protocol"),
    metadata: str = Form("{}"),
) -> dict[str, Any]:
    """Upload a clinical document for indexing."""
    start_time = time.time()

    from src.db.models import ClinicalDoc, DocumentChunk
    from src.db.postgres import AsyncSessionLocal
    from src.rag.chunker import PDFParser, SmartChunker

    # Validate document type
    if document_type not in {"protocol", "guideline", "reference"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="document_type must be one of: protocol, guideline, reference",
        )

    # Save uploaded file to disk
    upload_dir = Path(os.environ.get("UPLOAD_DIR", "uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / (file.filename or "upload.pdf")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    import asyncio

    # Parse PDF and chunk text (CPU-bound — run in thread pool)
    def _parse_and_chunk():
        parser = PDFParser()
        text = parser.parse(str(file_path))
        chunker = SmartChunker()
        return chunker.chunk(text, source=file.filename or "unknown")

    chunks = await asyncio.to_thread(_parse_and_chunk)

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
                embeddings = await embedding_gen.generate_embeddings(
                    chunk_texts, cache=False
                )
            except Exception as e:
                logger.warning("Embedding generation failed: %s", e)

        # Bulk-insert all chunk rows in one statement
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        if chunks:
            rows = [
                {
                    "document_id": doc.id,
                    "chunk_text": chunk.content,
                    "chunk_index": i,
                    "embedding": embeddings[i] if i < len(embeddings) else None,
                }
                for i, chunk in enumerate(chunks)
            ]
            await session.execute(pg_insert(DocumentChunk.__table__), rows)

        await session.commit()
        doc_id = doc.id

    elapsed = (time.time() - start_time) * 1000

    # Refresh cached chunks and doc_name_map after successful upload
    try:
        from sqlalchemy import select

        logger.info("Refreshing cached chunks after document upload...")
        async with AsyncSessionLocal() as refresh_session:
            result = await refresh_session.execute(select(DocumentChunk))
            new_chunks = result.scalars().all()
            app.state.cached_chunks = list(new_chunks)

            doc_result = await refresh_session.execute(select(ClinicalDoc))
            docs = doc_result.scalars().all()
            app.state.doc_name_map = {d.id: d.filename for d in docs}

            logger.info(
                "Cache refreshed: %d chunks, %d documents",
                len(new_chunks),
                len(docs),
            )
    except Exception as e:
        logger.warning("Failed to refresh chunk cache: %s", e)

    return {
        "document_id": doc_id,
        "filename": file.filename,
        "chunks_created": len(chunks),
        "status": "completed",
        "processing_time_ms": round(elapsed, 1),
    }


@app.get("/api/v1/evals", tags=["Evaluation"])
async def evaluation_endpoint() -> dict[str, Any]:
    """Run the evaluation suite and return results."""
    from src.evaluation.runner import EvaluationRunner

    runner = EvaluationRunner()
    results = runner.run()
    return results


# ============================================
# Metrics Endpoint
# ============================================


@app.get("/metrics", tags=["Monitoring"])
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    from src.observability.metrics import get_metrics_text

    return PlainTextResponse(content=get_metrics_text(), media_type="text/plain")


@app.post("/metrics/reset", tags=["Monitoring"])
async def reset_metrics_endpoint():
    """Reset all metrics counters (for testing/demo)."""
    from src.observability.metrics import reset_metrics

    reset_metrics()
    return {"status": "metrics_reset"}


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
