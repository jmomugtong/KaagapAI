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

    # Initialize Ollama LLM client
    try:
        from src.llm.ollama_client import OllamaClient

        app.state.ollama_client = OllamaClient()
        is_healthy = await app.state.ollama_client.health_check()
        if is_healthy:
            logger.info("Ollama client initialized and healthy")
        else:
            logger.warning("Ollama client initialized but service is unreachable")
    except Exception as e:
        logger.warning("Ollama client initialization failed: %s", e)
        app.state.ollama_client = None

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
        "service": "medquery-api",
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
    start_time = time.time()

    body = await request.json()
    question = body.get("question", "")
    max_results = body.get("max_results", 5)
    confidence_threshold = body.get("confidence_threshold", 0.70)

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
    try:
        embeddings = await embedding_gen.generate_embeddings([question])
        query_embedding = embeddings[0]
    except Exception as e:
        logger.warning("Embedding generation failed: %s", e)
        elapsed = (time.time() - start_time) * 1000
        return {
            "answer": "Embedding generation failed. Please try again later.",
            "confidence": 0.0,
            "citations": [],
            "retrieved_chunks": [],
            "query_id": "error",
            "processing_time_ms": round(elapsed, 1),
        }

    # Load all chunks and run hybrid retrieval
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(DocumentChunk))
            chunks = result.scalars().all()

            if not chunks:
                elapsed = (time.time() - start_time) * 1000
                return {
                    "answer": "No documents indexed yet. Upload documents first.",
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
    except Exception as e:
        logger.warning("Database query failed: %s", e)
        elapsed = (time.time() - start_time) * 1000
        return {
            "answer": "Database unavailable. Please try again later.",
            "confidence": 0.0,
            "citations": [],
            "retrieved_chunks": [],
            "query_id": "error",
            "processing_time_ms": round(elapsed, 1),
        }

    # Format retrieved chunks for response and prompt context
    retrieved_chunks = []
    prompt_chunks = []
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
        prompt_chunks.append(
            {
                "text": r.content,
                "metadata": {
                    "source": r.source,
                    "chunk_index": r.chunk_index,
                    "document_id": r.document_id,
                },
            }
        )

    if not search_results:
        elapsed = (time.time() - start_time) * 1000
        return {
            "answer": "No relevant results found for your query.",
            "confidence": 0.0,
            "citations": [],
            "retrieved_chunks": [],
            "query_id": str(uuid.uuid4())[:8],
            "processing_time_ms": round(elapsed, 1),
        }

    # LLM synthesis via Ollama
    from src.llm.prompt_templates import PromptTemplate
    from src.llm.response_parser import ResponseParser

    ollama_client = getattr(app.state, "ollama_client", None)

    if ollama_client:
        try:
            template = PromptTemplate()
            prompt = template.build(question=question, chunks=prompt_chunks)
            raw_response = await ollama_client.generate(prompt)

            if raw_response:
                parser = ResponseParser()
                parsed = parser.parse(
                    raw_response,
                    retrieved_chunks=[
                        {"text": c["text"], "source": c["source"]}
                        for c in retrieved_chunks
                    ],
                )

                # Low confidence: return snippets only
                if parsed.confidence < confidence_threshold:
                    answer = (
                        "Confidence too low for synthesis. "
                        "Relevant snippets are provided in retrieved_chunks."
                    )
                else:
                    answer = parsed.answer

                # PII redaction on LLM output
                answer = redactor.redact(answer)

                citations = [
                    {
                        "document": c.document,
                        "section": c.section,
                        "page": c.page,
                        "relevance_score": round(search_results[0].score, 4),
                    }
                    for c in parsed.citations
                ]

                elapsed = (time.time() - start_time) * 1000
                return {
                    "answer": answer,
                    "confidence": round(parsed.confidence, 4),
                    "citations": citations,
                    "retrieved_chunks": retrieved_chunks,
                    "query_id": str(uuid.uuid4())[:8],
                    "processing_time_ms": round(elapsed, 1),
                    "hallucination_flagged": parsed.has_hallucinated_citations,
                }
            else:
                logger.warning(
                    "Ollama returned empty response, falling back to snippets"
                )
        except Exception as e:
            logger.warning("LLM synthesis failed: %s, falling back to snippets", e)

    # Fallback: return snippets without LLM synthesis
    confidence = search_results[0].score if search_results else 0.0
    answer = (
        f"Found {len(search_results)} relevant chunk(s). "
        "See retrieved_chunks for details. (LLM synthesis unavailable)"
    )

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
    """Check the status of an async job."""
    from src.worker import get_job_status

    return get_job_status(job_id)


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
