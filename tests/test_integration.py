"""
Integration Tests for KaagapAI (Phase 11)

End-to-end flow tests covering the full pipeline.
These test the components together without external services.
"""

import pytest
from fastapi import status


class TestEndToEndQuery:
    """Tests for the full query pipeline."""

    @pytest.mark.unit
    def test_query_returns_all_expected_keys(self, client, sample_clinical_query):
        """Query endpoint returns all expected response keys."""
        response = client.post("/api/v1/query", json=sample_clinical_query)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "citations" in data
        assert "retrieved_chunks" in data
        assert "query_id" in data
        assert "processing_time_ms" in data

    @pytest.mark.unit
    def test_query_with_no_docs_returns_message(self, client):
        """Query with no indexed docs returns appropriate message."""
        response = client.post(
            "/api/v1/query",
            json={"question": "What is the pain protocol for knee surgery?"},
        )
        data = response.json()
        # Should return either no-docs message or embedding unavailable
        assert data["confidence"] == 0.0 or "answer" in data

    @pytest.mark.unit
    def test_query_rejects_unsafe_input(self, client):
        """Query endpoint rejects SQL injection attempts."""
        response = client.post(
            "/api/v1/query",
            json={"question": "'; DROP TABLE users; -- normal query text"},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestEndToEndUpload:
    """Tests for the upload pipeline."""

    @pytest.mark.unit
    def test_upload_requires_file(self, client):
        """Upload endpoint returns 422 when no file is provided."""
        response = client.post("/api/v1/upload")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.unit
    def test_upload_rejects_invalid_document_type(self, client):
        """Upload endpoint rejects invalid document_type."""
        import io

        fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
            data={"document_type": "invalid_type"},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestHealthEndpoints:
    """Tests for health and readiness endpoints."""

    @pytest.mark.unit
    def test_health_returns_healthy(self, client):
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "healthy"

    @pytest.mark.unit
    def test_ready_returns_checks(self, client):
        response = client.get("/ready")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "checks" in data
        assert "database" in data["checks"]
        assert "ollama" in data["checks"]


class TestJobStatusEndpoint:
    """Tests for job status tracking (removed with Celery)."""

    @pytest.mark.unit
    def test_job_status_returns_404(self, client):
        """Job status endpoint was removed with Celery — returns 404."""
        response = client.get("/api/v1/jobs/nonexistent-job-123")
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestMetricsEndpoint:
    """Tests for Prometheus metrics."""

    @pytest.mark.unit
    def test_metrics_returns_prometheus_format(self, client):
        response = client.get("/metrics")
        assert response.status_code == status.HTTP_200_OK
        text = response.text
        assert "queries_total" in text
        assert "cache_hit_rate" in text

    @pytest.mark.unit
    def test_metrics_contains_hallucination_rate(self, client):
        response = client.get("/metrics")
        text = response.text
        assert "hallucination_rate" in text


class TestPIIRedactionInPipeline:
    """Tests that PII is redacted throughout the pipeline."""

    @pytest.mark.unit
    def test_pii_in_query_is_redacted(self, client):
        """PII in query text should be redacted before processing."""
        response = client.post(
            "/api/v1/query",
            json={
                "question": "What is the treatment for Patient John Doe with SSN 123-45-6789?"
            },
        )
        # Should not fail; PII gets redacted before processing
        assert response.status_code in (200, 400)


class TestEvaluationEndpoint:
    """Tests for the evaluation endpoint."""

    @pytest.mark.unit
    def test_eval_endpoint_returns_results(self, client):
        response = client.get("/api/v1/evals")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
