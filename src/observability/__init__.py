"""
MedQuery Observability Module

Monitoring and logging components:
- Prometheus metrics
- Structured logging
"""

from src.observability.metrics import get_metrics_text, record_query

__all__ = ["get_metrics_text", "record_query"]
