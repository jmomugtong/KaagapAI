"""
Tests for MedQuery Celery Worker (Phase 9)
"""

import pytest

from src.worker import get_job_status


class TestJobStatus:
    """Tests for job status tracking."""

    @pytest.mark.unit
    def test_unknown_job_returns_status(self):
        result = get_job_status("nonexistent-job-123")
        assert "job_id" in result
        assert "status" in result

    @pytest.mark.unit
    def test_job_status_has_job_id(self):
        result = get_job_status("test-job-456")
        assert result["job_id"] == "test-job-456"
