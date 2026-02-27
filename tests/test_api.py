"""
Tests for KaagapAI API endpoints.
"""

import pytest
from fastapi import status


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.unit
    def test_health_check(self, client):
        """Test the /health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    @pytest.mark.unit
    def test_readiness_check(self, client):
        """Test the /ready endpoint returns ready status."""
        response = client.get("/ready")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ready"] is True
        assert "checks" in data


class TestQueryEndpoint:
    """Tests for the query endpoint."""

    @pytest.mark.unit
    def test_query_endpoint_returns_expected_keys(self, client, sample_clinical_query):
        """Test the query endpoint returns response with expected keys."""
        response = client.post("/api/v1/query", json=sample_clinical_query)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "citations" in data


class TestUploadEndpoint:
    """Tests for the document upload endpoint."""

    @pytest.mark.unit
    def test_upload_endpoint_requires_file(self, client):
        """Test the upload endpoint returns 422 when no file is provided."""
        response = client.post("/api/v1/upload")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestJobStatusEndpoint:
    """Tests for the job status endpoint (removed with Celery)."""

    @pytest.mark.unit
    def test_job_status_endpoint_returns_404(self, client):
        """Job status endpoint was removed with Celery — returns 404."""
        response = client.get("/api/v1/jobs/test-job-id")
        assert response.status_code == status.HTTP_404_NOT_FOUND
