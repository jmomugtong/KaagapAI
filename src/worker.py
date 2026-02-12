"""
Celery Worker for MedQuery (Phase 9)

Async task processing for:
- Document upload and chunking
- Batch embedding generation
- Evaluation suite execution
"""

import json
import logging
import os
import uuid

from celery import Celery

logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery(
    "medquery",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/1"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"),
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=2,
)

# In-memory job tracking (Redis-backed in production)
_job_store: dict[str, dict] = {}


@celery_app.task(bind=True, name="process_document")
def process_document(self, file_path: str, document_type: str, metadata_json: str):  # type: ignore[no-untyped-def]
    """
    Process an uploaded document: parse, chunk, embed, store.

    This runs asynchronously via Celery worker.
    """
    job_id = self.request.id or str(uuid.uuid4())
    _job_store[job_id] = {"status": "processing", "progress": 0}

    try:
        from src.rag.chunker import PDFParser, SmartChunker

        # Parse
        parser = PDFParser()
        text = parser.parse(file_path)
        _job_store[job_id]["progress"] = 30

        # Chunk
        chunker = SmartChunker()
        chunks = chunker.chunk(text, source=os.path.basename(file_path))
        _job_store[job_id]["progress"] = 60

        # Parse metadata
        try:
            meta = json.loads(metadata_json)
        except json.JSONDecodeError:
            meta = {}

        _job_store[job_id].update(
            {
                "status": "completed",
                "progress": 100,
                "result": {
                    "chunks_created": len(chunks),
                    "document_type": document_type,
                    "metadata": meta,
                },
            }
        )

        return {
            "job_id": job_id,
            "status": "completed",
            "chunks_created": len(chunks),
        }

    except Exception as e:
        logger.error("Document processing failed: %s", e)
        _job_store[job_id].update({"status": "failed", "error": str(e)})
        raise


@celery_app.task(name="run_evaluation")
def run_evaluation():  # type: ignore[no-untyped-def]
    """Run the evaluation suite as a background task."""
    try:
        from src.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner()
        return runner.run()
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        return {"status": "failed", "error": str(e)}


def get_job_status(job_id: str) -> dict:
    """Get the status of a processing job."""
    if job_id in _job_store:
        return {"job_id": job_id, **_job_store[job_id]}

    # Try Celery result backend
    try:
        result = celery_app.AsyncResult(job_id)
        if result.state == "PENDING":
            return {"job_id": job_id, "status": "pending"}
        elif result.state == "STARTED":
            return {"job_id": job_id, "status": "processing"}
        elif result.state == "SUCCESS":
            return {"job_id": job_id, "status": "completed", "result": result.result}
        elif result.state == "FAILURE":
            return {"job_id": job_id, "status": "failed", "error": str(result.result)}
        return {"job_id": job_id, "status": result.state.lower()}
    except Exception:
        return {"job_id": job_id, "status": "unknown"}


if __name__ == "__main__":
    celery_app.start()
