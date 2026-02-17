"""
Reranker for MedQuery

Dedicated cross-encoder reranking using FlashRank for sub-100ms
reranking on CPU. Falls back to retrieval ordering if FlashRank
is unavailable.

Replaces the previous LLM-based reranker that called Ollama per-chunk
(5 LLM calls per query = 15-40s). FlashRank reranks all chunks in a
single batch call in <100ms.
"""

import logging
from dataclasses import dataclass

from src.rag.retriever import ScoredChunk

logger = logging.getLogger(__name__)

# Confidence thresholds
HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.70


@dataclass
class RerankedChunk:
    """A chunk with both retrieval and rerank scores."""

    chunk_id: int
    content: str
    document_id: int
    chunk_index: int
    retrieval_score: float
    rerank_score: float
    final_score: float
    source: str


class Reranker:
    """FlashRank-based reranker for retrieved document chunks."""

    def __init__(self) -> None:
        self._ranker = None
        try:
            from flashrank import Ranker

            self._ranker = Ranker()
            logger.info("FlashRank reranker initialized")
        except Exception as e:
            logger.warning("FlashRank not available, using fallback: %s", e)

    async def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[RerankedChunk]:
        """
        Rerank chunks using FlashRank cross-encoder scoring.

        Falls back to original retrieval order if FlashRank is unavailable.
        """
        if not chunks:
            return []

        if self._ranker is None:
            return self._fallback_rerank(chunks, top_k)

        try:
            from flashrank import RerankRequest

            passages = [{"id": str(c.chunk_id), "text": c.content} for c in chunks]
            request = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(request)

            # Build a map from chunk_id to rerank score
            rerank_scores: dict[int, float] = {}
            for r in results:
                cid = int(r["id"])
                rerank_scores[cid] = float(r["score"])

            reranked = []
            for chunk in chunks:
                rr_score = rerank_scores.get(chunk.chunk_id, 0.5)
                # Blend: 0.3 * retrieval + 0.7 * rerank
                final_score = 0.3 * chunk.score + 0.7 * rr_score
                reranked.append(
                    RerankedChunk(
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        document_id=chunk.document_id,
                        chunk_index=chunk.chunk_index,
                        retrieval_score=chunk.score,
                        rerank_score=rr_score,
                        final_score=final_score,
                        source=chunk.source,  # Preserve original source (hybrid/bm25/vector)
                    )
                )

            reranked.sort(key=lambda x: x.final_score, reverse=True)
            return reranked[:top_k]

        except Exception as e:
            logger.warning("FlashRank reranking failed: %s", e)
            return self._fallback_rerank(chunks, top_k)

    def _fallback_rerank(
        self, chunks: list[ScoredChunk], top_k: int
    ) -> list[RerankedChunk]:
        """Fallback: convert ScoredChunks to RerankedChunks using retrieval score."""
        result = [
            RerankedChunk(
                chunk_id=c.chunk_id,
                content=c.content,
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                retrieval_score=c.score,
                rerank_score=c.score,
                final_score=c.score,
                source=c.source,  # Preserve original source (hybrid/bm25/vector)
            )
            for c in chunks
        ]
        result.sort(key=lambda x: x.final_score, reverse=True)
        return result[:top_k]


def assess_confidence(confidence: float) -> str:
    """Categorize confidence level for response strategy."""
    if confidence >= HIGH_CONFIDENCE:
        return "high"
    elif confidence >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"
