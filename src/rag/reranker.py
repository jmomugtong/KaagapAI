"""
Reranker for MedQuery (Phase 6)

LLM-based reranking of retrieved chunks by contextual relevance.
Uses Ollama to score each chunk against the query, then reorders
by relevance. Falls back to original ordering if Ollama is unavailable.
"""

import logging
import re
from dataclasses import dataclass

from src.llm.ollama_client import OllamaClient
from src.rag.retriever import ScoredChunk

logger = logging.getLogger(__name__)

RERANK_PROMPT = """Rate the relevance of the following document chunk to the query on a scale of 0.0 to 1.0.

QUERY: {query}

CHUNK: {chunk_text}

Respond with ONLY a single number between 0.0 and 1.0 representing the relevance score. Nothing else."""

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
    """LLM-based reranker for retrieved document chunks."""

    def __init__(self, ollama_client: OllamaClient | None = None) -> None:
        self._client = ollama_client

    async def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[RerankedChunk]:
        """
        Rerank chunks using LLM relevance scoring.

        Falls back to original retrieval order if LLM is unavailable.
        """
        if not chunks:
            return []

        if self._client is None:
            return self._fallback_rerank(chunks, top_k)

        reranked = []
        for chunk in chunks:
            rerank_score = await self._score_chunk(query, chunk.content)
            # Blend: 0.3 * retrieval + 0.7 * rerank
            final_score = 0.3 * chunk.score + 0.7 * rerank_score
            reranked.append(
                RerankedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    retrieval_score=chunk.score,
                    rerank_score=rerank_score,
                    final_score=final_score,
                    source="reranked",
                )
            )

        reranked.sort(key=lambda x: x.final_score, reverse=True)
        return reranked[:top_k]

    async def _score_chunk(self, query: str, chunk_text: str) -> float:
        """Score a single chunk's relevance to the query via LLM."""
        prompt = RERANK_PROMPT.format(query=query, chunk_text=chunk_text[:500])
        try:
            response = await self._client.generate(prompt)
            return self._parse_score(response)
        except Exception as e:
            logger.warning("Rerank scoring failed: %s", e)
            return 0.5

    def _parse_score(self, response: str) -> float:
        """Extract a float score from LLM response, clamped to [0.0, 1.0]."""
        match = re.search(r"([\d.]+)", response.strip())
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        return 0.5

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
                source="fallback",
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
