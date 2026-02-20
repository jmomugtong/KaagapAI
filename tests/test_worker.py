"""
Tests for MedQuery Celery Worker (Phase 9)
"""

import pytest

from src.worker import _job_store, celery_app, get_job_status, process_document, run_evaluation


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

    @pytest.mark.unit
    def test_job_in_store_returns_stored_data(self):
        """get_job_status returns stored data when job is in _job_store."""
        _job_store["store-test-001"] = {"status": "completed", "progress": 100}
        try:
            result = get_job_status("store-test-001")
            assert result["job_id"] == "store-test-001"
            assert result["status"] == "completed"
        finally:
            del _job_store["store-test-001"]


class TestCeleryResultFallback:
    """Tests for get_job_status Celery backend fallback branches."""

    @pytest.mark.unit
    def test_started_state_returns_processing(self, mocker):
        mock_result = mocker.MagicMock()
        mock_result.state = "STARTED"
        mocker.patch.object(celery_app, "AsyncResult", return_value=mock_result)
        result = get_job_status("celery-started-job")
        assert result["status"] == "processing"

    @pytest.mark.unit
    def test_success_state_returns_completed(self, mocker):
        mock_result = mocker.MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {"chunks_created": 5, "document_type": "protocol"}
        mocker.patch.object(celery_app, "AsyncResult", return_value=mock_result)
        result = get_job_status("celery-success-job")
        assert result["status"] == "completed"
        assert result["result"]["chunks_created"] == 5

    @pytest.mark.unit
    def test_failure_state_returns_failed(self, mocker):
        mock_result = mocker.MagicMock()
        mock_result.state = "FAILURE"
        mock_result.result = RuntimeError("Processing failed")
        mocker.patch.object(celery_app, "AsyncResult", return_value=mock_result)
        result = get_job_status("celery-failure-job")
        assert result["status"] == "failed"
        assert "error" in result

    @pytest.mark.unit
    def test_other_state_returns_lowercase_state(self, mocker):
        mock_result = mocker.MagicMock()
        mock_result.state = "RETRY"
        mocker.patch.object(celery_app, "AsyncResult", return_value=mock_result)
        result = get_job_status("celery-retry-job")
        assert result["status"] == "retry"


class TestProcessDocumentTask:
    """Tests for the process_document Celery task."""

    @pytest.mark.unit
    def test_process_document_success(self, mocker, tmp_path):
        """Task parses PDF, chunks it, and returns completed status."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        mock_parser = mocker.MagicMock()
        mock_parser.parse.return_value = "Sample clinical documentation text"
        mocker.patch("src.rag.chunker.PDFParser", return_value=mock_parser)

        mock_chunker = mocker.MagicMock()
        mock_chunker.chunk.return_value = [mocker.MagicMock(), mocker.MagicMock()]
        mocker.patch("src.rag.chunker.SmartChunker", return_value=mock_chunker)

        task_result = process_document.apply(
            args=[str(pdf_file), "protocol", '{"dept": "cardiology"}']
        )
        result = task_result.get()

        assert result["status"] == "completed"
        assert result["chunks_created"] == 2

    @pytest.mark.unit
    def test_process_document_invalid_json_metadata(self, mocker, tmp_path):
        """Task handles invalid metadata JSON gracefully (uses empty dict fallback)."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake content")

        mock_parser = mocker.MagicMock()
        mock_parser.parse.return_value = "Sample text"
        mocker.patch("src.rag.chunker.PDFParser", return_value=mock_parser)

        mock_chunker = mocker.MagicMock()
        mock_chunker.chunk.return_value = [mocker.MagicMock()]
        mocker.patch("src.rag.chunker.SmartChunker", return_value=mock_chunker)

        task_result = process_document.apply(
            args=[str(pdf_file), "guideline", "not-valid-json"]
        )
        result = task_result.get()

        assert result["status"] == "completed"
        assert result["chunks_created"] == 1

    @pytest.mark.unit
    def test_process_document_failure_marks_job_failed(self, mocker, tmp_path):
        """Task marks job as failed and re-raises when parsing errors."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake content")

        mocker.patch(
            "src.rag.chunker.PDFParser", side_effect=RuntimeError("Parse error")
        )

        task_result = process_document.apply(args=[str(pdf_file), "protocol", "{}"])
        assert task_result.failed()


class TestRunEvaluationTask:
    """Tests for the run_evaluation Celery task."""

    @pytest.mark.unit
    def test_run_evaluation_success(self, mocker):
        """Task returns runner output on success."""
        mock_runner = mocker.MagicMock()
        mock_runner.run.return_value = {"status": "passed", "rouge_l": 0.75}
        mocker.patch("src.evaluation.runner.EvaluationRunner", return_value=mock_runner)

        task_result = run_evaluation.apply()
        result = task_result.get()

        assert result["status"] == "passed"
        assert result["rouge_l"] == 0.75

    @pytest.mark.unit
    def test_run_evaluation_failure_returns_error_dict(self, mocker):
        """Task returns error dict (does not re-raise) when runner fails."""
        mocker.patch(
            "src.evaluation.runner.EvaluationRunner",
            side_effect=Exception("Eval exploded"),
        )

        task_result = run_evaluation.apply()
        result = task_result.get()

        assert result["status"] == "failed"
        assert "Eval exploded" in result["error"]
