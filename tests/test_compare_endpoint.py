"""
Tests for the Compare endpoint (/api/v1/compare)
"""

import pytest
from fastapi.testclient import TestClient


class TestCompareEndpoint:
    """Tests for POST /api/v1/compare."""

    @pytest.mark.unit
    def test_compare_returns_both_pipelines(self, client, mocker):
        """Compare endpoint returns classical and agentic results."""
        from src.pipelines.classical import PipelineResult

        mock_classical_result = PipelineResult(
            answer="Classical answer",
            confidence=0.75,
            citations=[],
            retrieved_chunks=[],
            query_id="c1",
            processing_time_ms=100.0,
            pipeline="classical",
            steps=[{"name": "retrieve", "duration_ms": 50, "detail": "ok"}],
        )
        mock_agentic_result = PipelineResult(
            answer="Agentic answer",
            confidence=0.90,
            citations=[],
            retrieved_chunks=[{"chunk_id": 1, "text": "a"}],
            query_id="a1",
            processing_time_ms=300.0,
            pipeline="agentic",
            steps=[
                {"name": "classify", "duration_ms": 100, "detail": "COMPARATIVE"},
                {"name": "retrieve", "duration_ms": 80, "detail": "Sub-query 1"},
                {"name": "retrieve", "duration_ms": 70, "detail": "Sub-query 2"},
            ],
        )

        mocker.patch(
            "src.pipelines.classical.ClassicalPipeline.run",
            return_value=mock_classical_result,
        )
        mocker.patch(
            "src.pipelines.agentic.AgenticPipeline.run",
            return_value=mock_agentic_result,
        )

        response = client.post(
            "/api/v1/compare",
            json={
                "question": "Compare pain protocols for knee vs hip replacement",
                "max_results": 3,
                "confidence_threshold": 0.70,
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert "classical" in data
        assert "agentic" in data
        assert "comparison" in data
        assert data["classical"]["pipeline"] == "classical"
        assert data["agentic"]["pipeline"] == "agentic"

    @pytest.mark.unit
    def test_compare_pipeline_fields_correct(self, client, mocker):
        """Both results have correct pipeline field."""
        from src.pipelines.classical import PipelineResult

        mock_result = PipelineResult(
            answer="test",
            confidence=0.5,
            citations=[],
            retrieved_chunks=[],
            query_id="q",
            processing_time_ms=50.0,
            pipeline="classical",
            steps=[],
        )
        mock_agentic_result = PipelineResult(
            answer="test",
            confidence=0.5,
            citations=[],
            retrieved_chunks=[],
            query_id="q",
            processing_time_ms=50.0,
            pipeline="agentic",
            steps=[],
        )

        mocker.patch(
            "src.pipelines.classical.ClassicalPipeline.run",
            return_value=mock_result,
        )
        mocker.patch(
            "src.pipelines.agentic.AgenticPipeline.run",
            return_value=mock_agentic_result,
        )

        response = client.post(
            "/api/v1/compare",
            json={"question": "What is the dosage for amoxicillin?"},
        )
        data = response.json()
        assert data["classical"]["pipeline"] == "classical"
        assert data["agentic"]["pipeline"] == "agentic"

    @pytest.mark.unit
    def test_compare_metrics_calculated(self, client, mocker):
        """Comparison metrics are computed correctly."""
        from src.pipelines.classical import PipelineResult

        mocker.patch(
            "src.pipelines.classical.ClassicalPipeline.run",
            return_value=PipelineResult(
                answer="a",
                confidence=0.60,
                citations=[],
                retrieved_chunks=[{"id": 1}, {"id": 2}],
                query_id="c",
                processing_time_ms=100.0,
                pipeline="classical",
                steps=[{"name": "retrieve", "duration_ms": 50, "detail": "ok"}],
            ),
        )
        mocker.patch(
            "src.pipelines.agentic.AgenticPipeline.run",
            return_value=PipelineResult(
                answer="b",
                confidence=0.88,
                citations=[],
                retrieved_chunks=[{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
                query_id="a",
                processing_time_ms=400.0,
                pipeline="agentic",
                steps=[
                    {"name": "retrieve", "duration_ms": 80, "detail": "sq1"},
                    {"name": "retrieve", "duration_ms": 70, "detail": "sq2"},
                ],
            ),
        )

        response = client.post(
            "/api/v1/compare",
            json={"question": "What is the dosage for amoxicillin?"},
        )
        comp = response.json()["comparison"]

        assert comp["latency_ratio"] == 4.0
        assert comp["confidence_delta"] == pytest.approx(0.28, abs=0.01)
        assert comp["classical_retrieval_passes"] == 1
        assert comp["agentic_retrieval_passes"] == 2
        assert comp["classical_chunks_used"] == 2
        assert comp["agentic_chunks_used"] == 4

    @pytest.mark.unit
    def test_compare_rejects_unsafe_input(self, client):
        """Input validation runs once before dispatch."""
        response = client.post(
            "/api/v1/compare",
            json={"question": "<script>alert('xss')</script> test question here"},
        )
        assert response.status_code == 400

    @pytest.mark.unit
    def test_compare_handles_pipeline_exception(self, client, mocker):
        """Error in one pipeline doesn't crash the endpoint."""
        from src.pipelines.classical import PipelineResult

        mocker.patch(
            "src.pipelines.classical.ClassicalPipeline.run",
            return_value=PipelineResult(
                answer="ok",
                confidence=0.8,
                citations=[],
                retrieved_chunks=[],
                query_id="c",
                processing_time_ms=100.0,
                pipeline="classical",
                steps=[],
            ),
        )
        mocker.patch(
            "src.pipelines.agentic.AgenticPipeline.run",
            side_effect=Exception("Agent crashed"),
        )

        response = client.post(
            "/api/v1/compare",
            json={"question": "What is the dosage for amoxicillin?"},
        )
        assert response.status_code == 200
        data = response.json()
        # Classical should succeed
        assert data["classical"]["answer"] == "ok"
        # Agentic should have error info
        assert "error" in data["agentic"]["answer"].lower()


class TestAgentQueryEndpoint:
    """Tests for POST /api/v1/agent/query."""

    @pytest.mark.unit
    def test_agent_query_returns_steps(self, client, mocker):
        """Agent query returns pipeline steps."""
        from src.pipelines.classical import PipelineResult

        mocker.patch(
            "src.pipelines.agentic.AgenticPipeline.run",
            return_value=PipelineResult(
                answer="Agent answer",
                confidence=0.85,
                citations=[],
                retrieved_chunks=[],
                query_id="a1",
                processing_time_ms=200.0,
                pipeline="agentic",
                steps=[
                    {"name": "classify", "duration_ms": 100, "detail": "SIMPLE"},
                    {"name": "complete", "duration_ms": 0, "detail": "Done"},
                ],
            ),
        )

        response = client.post(
            "/api/v1/agent/query",
            json={"question": "What is the dosage for amoxicillin?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline"] == "agentic"
        assert len(data["steps"]) == 2
        assert data["steps"][0]["name"] == "classify"

    @pytest.mark.unit
    def test_agent_query_rejects_unsafe_input(self, client):
        """Input validation blocks unsafe queries."""
        response = client.post(
            "/api/v1/agent/query",
            json={"question": "DROP TABLE users; -- long enough to pass length check"},
        )
        assert response.status_code == 400


class TestClassicalQueryBackwardCompat:
    """Verify existing /api/v1/query still works with refactored code."""

    @pytest.mark.unit
    def test_classical_query_returns_expected_fields(self, client, mocker):
        from src.pipelines.classical import PipelineResult

        mocker.patch(
            "src.pipelines.classical.ClassicalPipeline.run",
            return_value=PipelineResult(
                answer="Answer",
                confidence=0.9,
                citations=[],
                retrieved_chunks=[],
                query_id="q1",
                processing_time_ms=50.0,
                hallucination_flagged=False,
                pipeline="classical",
                steps=[],
            ),
        )

        response = client.post(
            "/api/v1/query",
            json={"question": "What is the dosage for amoxicillin?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "citations" in data
        assert "retrieved_chunks" in data
        assert "query_id" in data
        assert "processing_time_ms" in data
        # Should NOT have pipeline/steps fields (backward compat)
        assert "pipeline" not in data
        assert "steps" not in data
