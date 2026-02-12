"""
Tests for MedQuery Observability (Phase 13)
"""

import pytest

from src.observability.metrics import get_metrics_text, record_query


class TestMetrics:
    """Tests for Prometheus metrics."""

    @pytest.mark.unit
    def test_get_metrics_text_returns_string(self):
        text = get_metrics_text()
        assert isinstance(text, str)
        assert "queries_total" in text

    @pytest.mark.unit
    def test_record_query_increments_total(self):
        record_query(latency_ms=150.0, success=True)
        after = get_metrics_text()
        assert "queries_total" in after
        assert "queries_successful" in after

    @pytest.mark.unit
    def test_record_query_with_cache_hit(self):
        record_query(latency_ms=50.0, success=True, cache_hit=True)
        text = get_metrics_text()
        assert "cache_hit_rate" in text

    @pytest.mark.unit
    def test_record_query_with_hallucination(self):
        record_query(latency_ms=200.0, success=True, hallucination=True)
        text = get_metrics_text()
        assert "hallucination_rate" in text

    @pytest.mark.unit
    def test_metrics_contains_latency_percentiles(self):
        for i in range(10):
            record_query(latency_ms=100.0 + i * 50)
        text = get_metrics_text()
        assert "query_latency_seconds_p95" in text

    @pytest.mark.unit
    def test_record_failed_query(self):
        record_query(latency_ms=500.0, success=False)
        text = get_metrics_text()
        assert "queries_failed" in text
