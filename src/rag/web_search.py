"""
Web Search Fallback for KaagapAI

When no relevant local documents are found, falls back to DuckDuckGo
web search for medical information. Results are clearly marked as
web-sourced to distinguish from indexed clinical documents.

Uses the duckduckgo-search library (MIT license, zero API cost).
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MAX_WEB_RESULTS = 5


@dataclass
class WebResult:
    """A single web search result."""

    title: str
    snippet: str
    url: str
    source: str = "web"


async def search_web(query: str, max_results: int = MAX_WEB_RESULTS) -> list[WebResult]:
    """Search DuckDuckGo for medical information.

    Returns a list of WebResult with title, snippet, and URL.
    Falls back to empty list if duckduckgo-search is not installed
    or search fails.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning(
            "duckduckgo-search not installed. Install with: pip install duckduckgo-search"
        )
        return []

    try:
        # Add "medical" context to improve result quality
        medical_query = f"{query} medical clinical"

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(medical_query, max_results=max_results):
                results.append(
                    WebResult(
                        title=r.get("title", ""),
                        snippet=r.get("body", ""),
                        url=r.get("href", ""),
                    )
                )

        logger.info("Web search returned %d results for: %s", len(results), query)
        return results
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return []


def format_web_results_as_context(results: list[WebResult]) -> str:
    """Format web results into a context string for LLM synthesis.

    Clearly labels each result as web-sourced.
    """
    if not results:
        return ""

    sections = []
    for i, r in enumerate(results, 1):
        sections.append(
            f"[Web Source {i}: {r.title}]\n"
            f"URL: {r.url}\n"
            f"{r.snippet}"
        )

    return "\n\n".join(sections)


def web_results_to_chunks(results: list[WebResult]) -> list[dict]:
    """Convert web results to chunk-like dicts for pipeline compatibility."""
    return [
        {
            "chunk_id": f"web_{i}",
            "text": r.snippet,
            "document_id": f"web_{r.url}",
            "chunk_index": 0,
            "relevance_score": 0.5,  # Default score for web results
            "source": f"web: {r.title}",
            "url": r.url,
        }
        for i, r in enumerate(results)
    ]
