"""
Tests for KaagapAI Reranker (FlashRank-based)
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
    """Tests for reranker without FlashRank (fallback mode)."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fallback_preserves_order(self, mocker):
        mocker.patch(
            "src.rag.reranker.Reranker.__init__",
            lambda self: setattr(self, "_ranker", None),
        )
        chunks = [
            _make_chunk(1, "High relevance chunk", 0.9),
            _make_chunk(2, "Medium relevance chunk", 0.7),
            _make_chunk(3, "Low relevance chunk", 0.3),
        ]
        reranker = Reranker()
        results = await reranker.rerank("test query", chunks)
        assert len(results) == 3
        assert results[0].chunk_id == 1
        assert results[2].chunk_id == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fallback_returns_reranked_chunks(self, mocker):
        mocker.patch(
            "src.rag.reranker.Reranker.__init__",
            lambda self: setattr(self, "_ranker", None),
        )
        chunks = [_make_chunk(1, "text", 0.8)]
        reranker = Reranker()
        results = await reranker.rerank("query", chunks)
        assert isinstance(results[0], RerankedChunk)
        assert results[0].source == "hybrid"  # Preserves original chunk source

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fallback_respects_top_k(self, mocker):
        mocker.patch(
            "src.rag.reranker.Reranker.__init__",
            lambda self: setattr(self, "_ranker", None),
        )
        chunks = [_make_chunk(i, f"chunk {i}", 0.5) for i in range(10)]
        reranker = Reranker()
        results = await reranker.rerank("query", chunks, top_k=3)
        assert len(results) == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_chunks(self, mocker):
        mocker.patch(
            "src.rag.reranker.Reranker.__init__",
            lambda self: setattr(self, "_ranker", None),
        )
        reranker = Reranker()
        results = await reranker.rerank("query", [])
        assert results == []


class TestRerankerWithFlashRank:
    """Tests for reranker with mocked FlashRank."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rerank_with_flashrank_rescores(self, mocker):
        mock_ranker = mocker.MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": "1", "text": "Low retrieval but high relevance", "score": 0.95},
            {"id": "2", "text": "High retrieval but low relevance", "score": 0.40},
        ]
        mocker.patch(
            "src.rag.reranker.Reranker.__init__",
            lambda self: setattr(self, "_ranker", mock_ranker),
        )

        chunks = [
            _make_chunk(1, "Low retrieval but high relevance", 0.3),
            _make_chunk(2, "High retrieval but low relevance", 0.9),
        ]
        reranker = Reranker()
        results = await reranker.rerank("query", chunks)
        assert all(r.source == "hybrid" for r in results)  # Preserves original
        # Chunk 1: 0.3*0.3 + 0.7*0.95 = 0.755
        # Chunk 2: 0.3*0.9 + 0.7*0.40 = 0.55
        assert results[0].chunk_id == 1  # Higher final score after reranking

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rerank_batch_call(self, mocker):
        """Verify all chunks are passed in a single batch call."""
        mock_ranker = mocker.MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": "1", "text": "chunk 1", "score": 0.8},
            {"id": "2", "text": "chunk 2", "score": 0.6},
            {"id": "3", "text": "chunk 3", "score": 0.4},
        ]
        mocker.patch(
            "src.rag.reranker.Reranker.__init__",
            lambda self: setattr(self, "_ranker", mock_ranker),
        )

        chunks = [
            _make_chunk(1, "chunk 1", 0.5),
            _make_chunk(2, "chunk 2", 0.5),
            _make_chunk(3, "chunk 3", 0.5),
        ]
        reranker = Reranker()
        await reranker.rerank("query", chunks)
        # FlashRank should be called exactly once with all chunks
        assert mock_ranker.rerank.call_count == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rerank_handles_flashrank_failure(self, mocker):
        mock_ranker = mocker.MagicMock()
        mock_ranker.rerank.side_effect = Exception("FlashRank error")
        mocker.patch(
            "src.rag.reranker.Reranker.__init__",
            lambda self: setattr(self, "_ranker", mock_ranker),
        )

        chunks = [_make_chunk(1, "text", 0.8)]
        reranker = Reranker()
        results = await reranker.rerank("query", chunks)
        assert len(results) == 1
        assert results[0].source == "hybrid"  # Preserves original on fallback

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rerank_score_blending(self, mocker):
        """Verify the 0.3 retrieval + 0.7 rerank blending formula."""
        mock_ranker = mocker.MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": "1", "text": "text", "score": 0.80},
        ]
        mocker.patch(
            "src.rag.reranker.Reranker.__init__",
            lambda self: setattr(self, "_ranker", mock_ranker),
        )

        chunks = [_make_chunk(1, "text", 0.60)]
        reranker = Reranker()
        results = await reranker.rerank("query", chunks)
        # Expected: 0.3 * 0.60 + 0.7 * 0.80 = 0.18 + 0.56 = 0.74
        assert abs(results[0].final_score - 0.74) < 0.001
        assert abs(results[0].retrieval_score - 0.60) < 0.001
        assert abs(results[0].rerank_score - 0.80) < 0.001


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
