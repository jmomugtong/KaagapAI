"""
Prometheus Metrics for MedQuery (Phase 13)

Tracks:
- query_latency_seconds: Histogram of query response times
- cache_hit_rate: Gauge for embedding and query cache hit rates
- hallucination_rate: Gauge for hallucination detection rate
- queries_total: Counter of total queries processed
"""

import logging
import threading

logger = logging.getLogger(__name__)

# Thread-safe metrics storage
_lock = threading.Lock()

_metrics: dict[str, float] = {
    "queries_total": 0,
    "queries_successful": 0,
    "queries_failed": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "hallucinations_detected": 0,
    "avg_latency_ms": 0.0,
}

_latencies: list[float] = []


def record_query(
    latency_ms: float,
    success: bool = True,
    cache_hit: bool = False,
    hallucination: bool = False,
) -> None:
    """Record metrics for a processed query."""
    with _lock:
        _metrics["queries_total"] += 1
        if success:
            _metrics["queries_successful"] += 1
        else:
            _metrics["queries_failed"] += 1
        if cache_hit:
            _metrics["cache_hits"] += 1
        else:
            _metrics["cache_misses"] += 1
        if hallucination:
            _metrics["hallucinations_detected"] += 1
        _latencies.append(latency_ms)
        _metrics["avg_latency_ms"] = sum(_latencies) / len(_latencies)


def get_metrics_text() -> str:
    """Generate Prometheus-compatible metrics text."""
    with _lock:
        total = _metrics["queries_total"]
        cache_total = _metrics["cache_hits"] + _metrics["cache_misses"]
        cache_hit_rate = (
            _metrics["cache_hits"] / cache_total if cache_total > 0 else 0.0
        )
        hallucination_rate = (
            _metrics["hallucinations_detected"] / total if total > 0 else 0.0
        )

        # Compute percentile buckets
        sorted_latencies = sorted(_latencies) if _latencies else [0]
        p50 = _percentile(sorted_latencies, 50)
        p95 = _percentile(sorted_latencies, 95)
        p99 = _percentile(sorted_latencies, 99)

        lines = [
            "# HELP queries_total Total number of queries processed",
            "# TYPE queries_total counter",
            f'queries_total {int(_metrics["queries_total"])}',
            "",
            "# HELP queries_successful Total successful queries",
            "# TYPE queries_successful counter",
            f'queries_successful {int(_metrics["queries_successful"])}',
            "",
            "# HELP queries_failed Total failed queries",
            "# TYPE queries_failed counter",
            f'queries_failed {int(_metrics["queries_failed"])}',
            "",
            "# HELP query_latency_seconds Query response time histogram",
            "# TYPE query_latency_seconds histogram",
            f'query_latency_seconds{{le="0.5"}} {_count_below(sorted_latencies, 500)}',
            f'query_latency_seconds{{le="1.0"}} {_count_below(sorted_latencies, 1000)}',
            f'query_latency_seconds{{le="2.0"}} {_count_below(sorted_latencies, 2000)}',
            f'query_latency_seconds{{le="5.0"}} {_count_below(sorted_latencies, 5000)}',
            f"query_latency_seconds_p50 {p50 / 1000:.4f}",
            f"query_latency_seconds_p95 {p95 / 1000:.4f}",
            f"query_latency_seconds_p99 {p99 / 1000:.4f}",
            "",
            "# HELP cache_hit_rate Cache hit ratio",
            "# TYPE cache_hit_rate gauge",
            f"cache_hit_rate {cache_hit_rate:.4f}",
            "",
            "# HELP hallucination_rate Rate of hallucinated responses",
            "# TYPE hallucination_rate gauge",
            f"hallucination_rate {hallucination_rate:.4f}",
        ]

        return "\n".join(lines) + "\n"


def reset_metrics() -> None:
    """Reset all metrics to zero."""
    with _lock:
        for key in _metrics:
            _metrics[key] = 0
        _latencies.clear()


def _percentile(sorted_data: list[float], percentile: int) -> float:
    """Compute the given percentile from sorted data."""
    if not sorted_data:
        return 0.0
    idx = int(len(sorted_data) * percentile / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def _count_below(sorted_data: list[float], threshold_ms: float) -> int:
    """Count values below threshold in sorted data."""
    count = 0
    for v in sorted_data:
        if v <= threshold_ms:
            count += 1
        else:
            break
    return count
