"""
Tests for MedQuery Reranker (Phase 6)
"""

import pytest

from src.rag.reranker import RerankedChunk, Reranker, assess_confidence
from src.rag.retriever import ScoredChunk


def _make_chunk(chunk_id: int, content: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        content=content,
        document_id=1,
        chunk_index=chunk_id,
        score=score,
        source="hybrid",
    )


class TestRerankerFallback:
    """Tests for reranker without LLM (fallback mode)."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fallback_preserves_order(self):
        chunks = [
            _make_chunk(1, "High relevance chunk", 0.9),
            _make_chunk(2, "Medium relevance chunk", 0.7),
            _make_chunk(3, "Low relevance chunk", 0.3),
        ]
        reranker = Reranker(ollama_client=None)
        results = await reranker.rerank("test query", chunks)
        assert len(results) == 3
        assert results[0].chunk_id == 1
        assert results[2].chunk_id == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fallback_returns_reranked_chunks(self):
        chunks = [_make_chunk(1, "text", 0.8)]
        reranker = Reranker(ollama_client=None)
        results = await reranker.rerank("query", chunks)
        assert isinstance(results[0], RerankedChunk)
        assert results[0].source == "fallback"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fallback_respects_top_k(self):
        chunks = [_make_chunk(i, f"chunk {i}", 0.5) for i in range(10)]
        reranker = Reranker(ollama_client=None)
        results = await reranker.rerank("query", chunks, top_k=3)
        assert len(results) == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        reranker = Reranker(ollama_client=None)
        results = await reranker.rerank("query", [])
        assert results == []


class TestRerankerWithLLM:
    """Tests for reranker with mocked LLM."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rerank_with_llm_rescores(self, mocker):
        mock_client = mocker.AsyncMock()
        mock_client.generate.return_value = "0.95"

        chunks = [
            _make_chunk(1, "Low retrieval but high relevance", 0.3),
            _make_chunk(2, "High retrieval but low relevance", 0.9),
        ]
        reranker = Reranker(ollama_client=mock_client)
        results = await reranker.rerank("query", chunks)
        assert all(r.source == "reranked" for r in results)
        assert all(r.rerank_score == 0.95 for r in results)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rerank_handles_llm_failure(self, mocker):
        mock_client = mocker.AsyncMock()
        mock_client.generate.side_effect = Exception("LLM error")

        chunks = [_make_chunk(1, "text", 0.8)]
        reranker = Reranker(ollama_client=mock_client)
        results = await reranker.rerank("query", chunks)
        assert len(results) == 1
        assert results[0].rerank_score == 0.5  # fallback score


class TestRerankerScoreParsing:
    """Tests for score parsing from LLM responses."""

    @pytest.mark.unit
    def test_parse_clean_score(self):
        reranker = Reranker()
        assert reranker._parse_score("0.85") == 0.85

    @pytest.mark.unit
    def test_parse_score_with_text(self):
        reranker = Reranker()
        assert reranker._parse_score("The relevance is 0.72") == 0.72

    @pytest.mark.unit
    def test_parse_score_clamped_high(self):
        reranker = Reranker()
        assert reranker._parse_score("1.5") == 1.0

    @pytest.mark.unit
    def test_parse_score_invalid(self):
        reranker = Reranker()
        assert reranker._parse_score("not a number") == 0.5


class TestConfidenceAssessment:
    """Tests for confidence level categorization."""

    @pytest.mark.unit
    def test_high_confidence(self):
        assert assess_confidence(0.90) == "high"

    @pytest.mark.unit
    def test_medium_confidence(self):
        assert assess_confidence(0.75) == "medium"

    @pytest.mark.unit
    def test_low_confidence(self):
        assert assess_confidence(0.50) == "low"

    @pytest.mark.unit
    def test_boundary_high(self):
        assert assess_confidence(0.85) == "high"

    @pytest.mark.unit
    def test_boundary_medium(self):
        assert assess_confidence(0.70) == "medium"
