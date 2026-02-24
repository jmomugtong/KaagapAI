"""
Extended tests for MedQuery API endpoints — covering lines not reached by test_api.py.

Target coverage: /ready degraded paths, /api/v1/agent/query, /api/v1/compare,
/api/v1/upload (success + bad type), /api/v1/jobs (unknown), /api/v1/evals.
"""

import io

import pytest
from fastapi import status

from src.pipelines.classical import PipelineResult


def _make_result(**kwargs):
    """Build a minimal PipelineResult for mocking."""
    defaults = dict(
        answer="Test answer",
        confidence=0.85,
        citations=[],
        retrieved_chunks=[],
        query_id="q123",
        processing_time_ms=100.0,
        hallucination_flagged=False,
        cached=False,
        pipeline="classical",
        steps=[],
    )
    defaults.update(kwargs)
    return PipelineResult(**defaults)


# ============================================
# /ready endpoint
# ============================================


class TestReadyEndpoint:
    """Detailed tests for the /ready readiness endpoint."""

    @pytest.mark.unit
    def test_ready_ollama_unavailable(self, client, mocker):
        """When no ollama_client is set, ollama status should be 'unavailable'."""
        from src.main import app

        # Ensure ollama_client is not set
        original = getattr(app.state, "ollama_client", "SENTINEL")
        app.state.ollama_client = None
        try:
            response = client.get("/ready")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["checks"]["ollama"] == "unavailable"
        finally:
            if original == "SENTINEL":
                del app.state.ollama_client
            else:
                app.state.ollama_client = original

    @pytest.mark.unit
    def test_ready_ollama_health_check_raises(self, client, mocker):
        """When ollama health_check raises, status should be 'error'."""
        from src.main import app

        mock_client = mocker.AsyncMock()
        mock_client.health_check.side_effect = Exception("connection refused")
        original = getattr(app.state, "ollama_client", "SENTINEL")
        app.state.ollama_client = mock_client
        try:
            response = client.get("/ready")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["checks"]["ollama"] == "error"
        finally:
            if original == "SENTINEL":
                del app.state.ollama_client
            else:
                app.state.ollama_client = original

    @pytest.mark.unit
    def test_ready_ollama_degraded(self, client, mocker):
        """When health_check returns False, ollama status should be 'degraded'."""
        from src.main import app

        mock_client = mocker.AsyncMock()
        mock_client.health_check.return_value = False
        original = getattr(app.state, "ollama_client", "SENTINEL")
        app.state.ollama_client = mock_client
        try:
            response = client.get("/ready")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["checks"]["ollama"] == "degraded"
        finally:
            if original == "SENTINEL":
                del app.state.ollama_client
            else:
                app.state.ollama_client = original

    @pytest.mark.unit
    def test_ready_embedding_unavailable(self, client):
        """When no embedding_generator is set, embedding_model should be 'unavailable'."""
        from src.main import app

        original = getattr(app.state, "embedding_generator", "SENTINEL")
        app.state.embedding_generator = None
        try:
            response = client.get("/ready")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["checks"]["embedding_model"] == "unavailable"
        finally:
            if original == "SENTINEL":
                del app.state.embedding_generator
            else:
                app.state.embedding_generator = original


# ============================================
# /api/v1/agent/query endpoint
# ============================================


class TestAgentQueryEndpoint:
    """Tests for the agentic RAG query endpoint."""

    @pytest.mark.unit
    def test_agent_query_success(self, client, mocker):
        """Successful agentic query returns expected keys."""
        agentic_result = _make_result(
            answer="Agentic answer",
            confidence=0.88,
            pipeline="agentic",
            steps=[{"name": "classify", "duration_ms": 5, "detail": "SIMPLE"}],
        )
        mocker.patch(
            "src.pipelines.agentic.AgenticPipeline.run",
            return_value=agentic_result,
        )
        response = client.post(
            "/api/v1/agent/query",
            json={"question": "What is the dosage for metformin?"},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["answer"] == "Agentic answer"
        assert data["pipeline"] == "agentic"
        assert "steps" in data
        assert "confidence" in data

    @pytest.mark.unit
    def test_agent_query_unsafe_input_returns_400(self, client):
        """Unsafe input is rejected with 400."""
        response = client.post(
            "/api/v1/agent/query",
            json={"question": "x' OR '1'='1"},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.unit
    def test_agent_query_xss_input_returns_400(self, client):
        """XSS/script injection input is rejected with 400."""
        response = client.post(
            "/api/v1/agent/query",
            json={"question": "<script>alert('xss')</script>"},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.unit
    def test_agent_query_pipeline_returns_low_confidence(self, client, mocker):
        """Low-confidence agentic result is still returned (no error)."""
        agentic_result = _make_result(
            answer="Confidence too low",
            confidence=0.30,
            pipeline="agentic",
        )
        mocker.patch(
            "src.pipelines.agentic.AgenticPipeline.run",
            return_value=agentic_result,
        )
        response = client.post(
            "/api/v1/agent/query",
            json={"question": "What is the treatment for Type 2 diabetes?"},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["confidence"] == 0.30


# ============================================
# /api/v1/compare endpoint
# ============================================


class TestCompareEndpoint:
    """Tests for the side-by-side pipeline comparison endpoint."""

    @pytest.mark.unit
    def test_compare_success(self, client, mocker):
        """Successful compare returns both classical and agentic results."""
        classical_result = _make_result(
            answer="Classical answer", confidence=0.80, pipeline="classical"
        )
        agentic_result = _make_result(
            answer="Agentic answer", confidence=0.90, pipeline="agentic"
        )
        mocker.patch(
            "src.pipelines.classical.ClassicalPipeline.run",
            return_value=classical_result,
        )
        mocker.patch(
            "src.pipelines.agentic.AgenticPipeline.run",
            return_value=agentic_result,
        )
        response = client.post(
            "/api/v1/compare",
            json={"question": "What are the contraindications for ACE inhibitors?"},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "classical" in data
        assert "agentic" in data
        assert "comparison" in data
        assert data["classical"]["answer"] == "Classical answer"
        assert data["agentic"]["answer"] == "Agentic answer"
        assert "confidence_delta" in data["comparison"]

    @pytest.mark.unit
    def test_compare_unsafe_input_returns_400(self, client):
        """Unsafe input to compare endpoint returns 400."""
        response = client.post(
            "/api/v1/compare",
            json={"question": "x' OR '1'='1"},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.unit
    def test_compare_pipeline_exception_returns_error_dict(self, client, mocker):
        """If one pipeline raises, compare still returns a response with error info."""
        mocker.patch(
            "src.pipelines.classical.ClassicalPipeline.run",
            side_effect=Exception("classical boom"),
        )
        mocker.patch(
            "src.pipelines.agentic.AgenticPipeline.run",
            side_effect=Exception("agentic boom"),
        )
        response = client.post(
            "/api/v1/compare",
            json={"question": "What is the treatment for Type 2 diabetes?"},
        )
        # asyncio.gather(return_exceptions=True) means the endpoint returns 200
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "Pipeline error" in data["classical"]["answer"]
        assert "Pipeline error" in data["agentic"]["answer"]


# ============================================
# /api/v1/upload endpoint
# ============================================


class TestUploadEndpointExtended:
    """Extended upload endpoint tests — success path and validation."""

    @pytest.mark.unit
    def test_upload_invalid_document_type_returns_400(self, client):
        """Uploading with an invalid document_type returns 400."""
        pdf_bytes = b"%PDF-1.4 fake"
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"document_type": "invalid_type"},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "document_type" in response.json()["detail"]

    @pytest.mark.unit
    def test_upload_success_mock(self, client, mocker, tmp_path):
        """Successful upload returns chunks_created and status=completed."""
        import asyncio as _asyncio
        from unittest.mock import MagicMock

        from src.rag.chunker import Chunk

        # Mock PDF parsing and chunking (asyncio.to_thread wraps a sync fn)
        mock_chunks = [
            Chunk(content="Chunk 1 text", metadata={"source": "report.pdf"}),
            Chunk(content="Chunk 2 text", metadata={"source": "report.pdf"}),
        ]
        mocker.patch.object(
            _asyncio,
            "to_thread",
            new=mocker.AsyncMock(return_value=mock_chunks),
        )

        # Mock DB session
        mock_session = mocker.AsyncMock()
        mock_session.add = MagicMock(side_effect=lambda obj: setattr(obj, "id", 42))
        mock_session.flush = mocker.AsyncMock()
        mock_session.execute = mocker.AsyncMock()
        mock_session.commit = mocker.AsyncMock()
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=False)
        mocker.patch("src.db.postgres.AsyncSessionLocal", return_value=mock_session)

        # Mock embedding generator on app.state
        from src.main import app

        original_emb = getattr(app.state, "embedding_generator", "SENTINEL")
        mock_emb = mocker.AsyncMock()
        mock_emb.generate_embeddings.return_value = [[0.1] * 768, [0.2] * 768]
        app.state.embedding_generator = mock_emb

        try:
            pdf_bytes = b"%PDF-1.4 fake content"
            response = client.post(
                "/api/v1/upload",
                files={"file": ("report.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
                data={"document_type": "protocol", "metadata": "{}"},
            )
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "chunks_created" in data
            assert data["chunks_created"] == 2
            assert data["status"] == "completed"
        finally:
            if original_emb == "SENTINEL":
                try:
                    del app.state.embedding_generator
                except AttributeError:
                    pass
            else:
                app.state.embedding_generator = original_emb


# ============================================
# /api/v1/jobs/{job_id} endpoint
# ============================================


class TestJobStatusEndpointExtended:
    """Extended job status endpoint tests."""

    @pytest.mark.unit
    def test_job_status_unknown_id(self, client):
        """Unknown job_id returns a valid response (not 404 — worker returns status dict)."""
        response = client.get("/api/v1/jobs/nonexistent-job-id-xyz")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # get_job_status returns a dict with job_id even for unknown IDs
        assert "job_id" in data


# ============================================
# /api/v1/evals endpoint
# ============================================


class TestEvalsEndpoint:
    """Tests for the evaluation endpoint."""

    @pytest.mark.unit
    def test_evals_no_dataset(self, client, mocker):
        """When no dataset exists, evals returns no_dataset status."""
        mocker.patch(
            "src.evaluation.runner.EvaluationRunner.load_dataset",
            return_value=[],
        )
        response = client.get("/api/v1/evals")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] in ("no_dataset", "no_results")

    @pytest.mark.unit
    def test_evals_with_dataset_runs_pipeline(self, client, mocker):
        """When dataset is present, evals runs and returns metrics."""
        questions = [
            {
                "id": "q001",
                "query": "What is the first-line treatment for hypertension?",
                "ground_truth": "ACE inhibitors or ARBs are first-line treatments.",
                "expected_sources": ["Hypertension Guidelines v2"],
            }
        ]
        mocker.patch(
            "src.evaluation.runner.EvaluationRunner.load_dataset",
            return_value=questions,
        )
        # Mock the pipeline run inside the runner
        mock_result = _make_result(
            answer="ACE inhibitors are first-line treatments for hypertension.",
            confidence=0.85,
            retrieved_chunks=[{"source": "Hypertension Guidelines v2", "text": "..."}],
            hallucination_flagged=False,
        )
        mocker.patch(
            "src.evaluation.runner._run_pipeline_for_question",
            return_value=mock_result,
        )
        response = client.get("/api/v1/evals")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "completed"
        assert "rouge_l_avg" in data["metrics"]
        assert "hallucination_rate" in data["metrics"]
        assert "retrieval_recall" in data["metrics"]


# ============================================
# Rate limiting
# ============================================


class TestRateLimiting:
    """Verify rate-limit response headers and 429 behavior."""

    @pytest.mark.unit
    def test_rate_limit_exceeded_returns_429(self):
        """After 10 requests/min, subsequent requests return 429."""
        from src.security.rate_limiter import RateLimiter

        limiter = RateLimiter()
        key = "rate:testuser"

        # Make 10 allowed requests
        for _ in range(10):
            allowed, _ = limiter.is_allowed(key)
            assert allowed is True

        # The 11th should be rate-limited
        allowed, retry_after = limiter.is_allowed(key)
        assert allowed is False
        assert retry_after > 0
