"""
Tests for MedQuery API endpoints.
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
    def test_query_endpoint_placeholder(self, client, sample_clinical_query):
        """Test the query endpoint returns placeholder response."""
        response = client.post("/api/v1/query", json=sample_clinical_query)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "citations" in data


class TestUploadEndpoint:
    """Tests for the document upload endpoint."""

    @pytest.mark.unit
    def test_upload_endpoint_placeholder(self, client):
        """Test the upload endpoint returns placeholder response."""
        response = client.post("/api/v1/upload")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data


class TestJobStatusEndpoint:
    """Tests for the job status endpoint."""

    @pytest.mark.unit
    def test_job_status_endpoint(self, client):
        """Test the job status endpoint."""
        response = client.get("/api/v1/jobs/test-job-id")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["job_id"] == "test-job-id"
