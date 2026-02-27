"""
Reranker for KaagapAI

Dedicated cross-encoder reranking using FlashRank for sub-100ms
reranking on CPU. Falls back to retrieval ordering if FlashRank
is unavailable.

Enhanced with sentence-level extraction: after chunk-level reranking,
extracts the most relevant sentences from top chunks for more focused
LLM context.
"""

import logging
import re
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
    document_name: str = ""


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
                # Blend: 0.5 * retrieval + 0.5 * rerank (equal weight — FlashRank is general-purpose, not clinical-specific)
                final_score = 0.5 * chunk.score + 0.5 * rr_score
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
                        document_name=chunk.document_name,
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
                document_name=c.document_name,
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


# ============================================
# Sentence-Level Extraction
# ============================================

# Sentence boundary regex (handles abbreviations better than split("."))
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])" r"|(?<=\n)\s*(?=\S)")

# Common abbreviations that should NOT trigger sentence splits
_ABBREVIATIONS = [
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "al.",
    "approx.",
    "dept.",
    "vol.",
]
_ABBR_PLACEHOLDER = "\x00"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling medical text patterns."""
    # Protect abbreviations from being treated as sentence boundaries
    protected = text.strip()
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _ABBR_PLACEHOLDER))

    sentences = SENTENCE_BOUNDARY.split(protected)

    # Restore abbreviations and filter out tiny fragments
    result = []
    for s in sentences:
        restored = s.strip().replace(_ABBR_PLACEHOLDER, ".")
        if len(restored) > 20:
            result.append(restored)
    return result


def extract_key_sentences(
    chunks: list[RerankedChunk],
    query: str,
    max_sentences: int = 10,
) -> list[tuple[str, str]]:
    """Extract the most relevant sentences from reranked chunks.

    Uses BM25 at the sentence level to pick the most query-relevant
    sentences from across all top chunks.

    Returns list of (sentence, document_name) tuples.
    """
    if not chunks:
        return []

    # Collect all sentences with their source chunk score and document name
    all_sentences: list[tuple[str, float, str]] = []
    for chunk in chunks:
        doc_name = chunk.document_name or f"Document {chunk.document_id}"
        sentences = _split_sentences(chunk.content)
        for sent in sentences:
            all_sentences.append((sent, chunk.final_score, doc_name))

    if not all_sentences:
        return [
            (c.content, c.document_name or f"Document {c.document_id}")
            for c in chunks[:3]
        ]

    # Use BM25 to rank sentences by relevance to query
    try:
        from rank_bm25 import BM25Okapi

        corpus = [s[0].lower().split() for s in all_sentences]
        if not corpus or all(len(doc) == 0 for doc in corpus):
            return [(s[0], s[2]) for s in all_sentences[:max_sentences]]

        bm25 = BM25Okapi(corpus)
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        # Combine BM25 sentence score with chunk-level score
        scored = []
        for i, (sent, chunk_score, doc_name) in enumerate(all_sentences):
            combined = 0.4 * (scores[i] / max(max(scores), 1.0)) + 0.6 * chunk_score
            scored.append((sent, combined, doc_name))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(s[0], s[2]) for s in scored[:max_sentences]]
    except Exception as e:
        logger.warning("Sentence-level extraction failed: %s", e)
        return [(s[0], s[2]) for s in all_sentences[:max_sentences]]


def build_extractive_answer(
    chunks: list[RerankedChunk],
    query: str,
    max_sentences: int = 5,
) -> str:
    """Build an extractive answer from the top sentences of reranked chunks.

    Used as a fallback when LLM confidence is low — returns real text
    from the documents rather than generated content.
    """
    sentences = extract_key_sentences(chunks, query, max_sentences=max_sentences)
    if not sentences:
        return "No relevant information found in the indexed documents."

    parts = []
    for i, (sent, doc_name) in enumerate(sentences, 1):
        parts.append(f"{i}. {sent} [{doc_name}]")

    return (
        "Based on the most relevant passages from indexed documents:\n\n"
        + "\n\n".join(parts)
    )
