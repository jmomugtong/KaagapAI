"""
Hybrid Retrieval System for MedQuery

Combines BM25 keyword search with pgvector cosine similarity search
using a weighted fusion strategy: 0.4 * BM25 + 0.6 * cosine.
"""

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ============================================
# Medical Abbreviation Dictionary
# ============================================

MEDICAL_ABBREVIATIONS: dict[str, str] = {
    "MI": "myocardial infarction",
    "CHF": "congestive heart failure",
    "DVT": "deep vein thrombosis",
    "PE": "pulmonary embolism",
    "COPD": "chronic obstructive pulmonary disease",
    "HTN": "hypertension",
    "DM": "diabetes mellitus",
    "CAD": "coronary artery disease",
    "CABG": "coronary artery bypass graft",
    "CVA": "cerebrovascular accident",
    "TIA": "transient ischemic attack",
    "AFib": "atrial fibrillation",
    "ACS": "acute coronary syndrome",
    "STEMI": "st elevation myocardial infarction",
    "NSTEMI": "non st elevation myocardial infarction",
    "CKD": "chronic kidney disease",
    "AKI": "acute kidney injury",
    "ARDS": "acute respiratory distress syndrome",
    "ICU": "intensive care unit",
    "OR": "operating room",
}

STOP_WORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "me", "my", "no", "nor",
    "not", "of", "on", "or", "our", "out", "own", "she", "so",
    "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "to", "too", "up", "us", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your",
}


# ============================================
# ScoredChunk
# ============================================


@dataclass
class ScoredChunk:
    """A document chunk with a relevance score."""

    chunk_id: int
    content: str
    document_id: int
    chunk_index: int
    score: float
    source: str  # "bm25", "vector", or "hybrid"


# ============================================
# QueryPreprocessor
# ============================================


class QueryPreprocessor:
    """Preprocesses queries with normalization, abbreviation expansion, and tokenization."""

    def __init__(self):
        self._abbreviations = MEDICAL_ABBREVIATIONS
        self._stop_words = STOP_WORDS

    def preprocess(self, query: str) -> str:
        """Lowercase and expand medical abbreviations."""
        result = self._expand_abbreviations(query)
        return result.lower()

    def _expand_abbreviations(self, text: str) -> str:
        """Replace medical abbreviations with full terms."""
        for abbr, expansion in self._abbreviations.items():
            # Match whole word only (case-sensitive for abbreviations)
            pattern = r"\b" + re.escape(abbr) + r"\b"
            text = re.sub(pattern, expansion, text)
        return text

    def tokenize(self, query: str) -> list[str]:
        """Preprocess, tokenize, and remove stop words."""
        processed = self.preprocess(query)
        tokens = processed.split()
        return [t for t in tokens if t not in self._stop_words]


# ============================================
# BM25Retriever
# ============================================


class BM25Retriever:
    """BM25 keyword search over document chunks."""

    def __init__(self, chunks: list):
        self._chunks = chunks
        self._preprocessor = QueryPreprocessor()
        self._index = None
        self._corpus_tokens: list[list[str]] = []
        self._build_index()

    def _build_index(self) -> None:
        """Build BM25 index from chunk contents."""
        if not self._chunks:
            return
        self._corpus_tokens = [
            self._preprocessor.tokenize(chunk.content) for chunk in self._chunks
        ]
        self._index = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        """Search for chunks matching the query keywords."""
        if not self._chunks or self._index is None:
            return []

        query_tokens = self._preprocessor.tokenize(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)

        # Normalize scores to 0.0-1.0
        max_score = max(scores) if max(scores) > 0 else 1.0
        normalized = [s / max_score for s in scores]

        # Pair with chunks and filter zeros
        scored = []
        for chunk, score in zip(self._chunks, normalized, strict=True):
            if score > 0:
                scored.append(
                    ScoredChunk(
                        chunk_id=chunk.id,
                        content=chunk.content,
                        document_id=chunk.document_id,
                        chunk_index=chunk.chunk_index,
                        score=score,
                        source="bm25",
                    )
                )

        # Sort descending and return top_k
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


# ============================================
# VectorRetriever
# ============================================


class VectorRetriever:
    """Vector similarity search using pgvector cosine distance."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[ScoredChunk]:
        """Search for chunks by cosine similarity to query embedding."""
        try:
            embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
            sql = text(
                "SELECT id, chunk_text, document_id, chunk_index, "
                "1 - (embedding <=> :query_vector) AS similarity "
                "FROM embeddings_cache "
                "ORDER BY embedding <=> :query_vector "
                "LIMIT :top_k"
            )
            result = await self._session.execute(
                sql, {"query_vector": embedding_str, "top_k": top_k}
            )
            rows = result.fetchall()

            return [
                ScoredChunk(
                    chunk_id=row.id,
                    content=row.chunk_text,
                    document_id=row.document_id,
                    chunk_index=row.chunk_index,
                    score=float(row.similarity),
                    source="vector",
                )
                for row in rows
            ]
        except Exception:
            logger.warning("Vector search failed, returning empty results", exc_info=True)
            return []


# ============================================
# HybridRetriever
# ============================================

BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6


class HybridRetriever:
    """Combines BM25 and vector search with weighted fusion scoring."""

    def __init__(self, chunks: list, session: AsyncSession):
        self._bm25 = BM25Retriever(chunks)
        self._vector = VectorRetriever(session)

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        """
        Hybrid search combining BM25 and vector similarity.

        Fusion: final_score = 0.4 * bm25_score + 0.6 * cosine_similarity
        Deduplicates by chunk_id, returns top_k results sorted by score.
        """
        # Get results from both retrievers
        bm25_results = self._bm25.search(query, top_k=10)
        vector_results = await self._vector.search(query_embedding, top_k=10)

        # Merge into a dict keyed by chunk_id
        merged: dict[int, dict] = {}

        for chunk in bm25_results:
            merged[chunk.chunk_id] = {
                "content": chunk.content,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "bm25_score": chunk.score,
                "vector_score": 0.0,
            }

        for chunk in vector_results:
            if chunk.chunk_id in merged:
                merged[chunk.chunk_id]["vector_score"] = chunk.score
            else:
                merged[chunk.chunk_id] = {
                    "content": chunk.content,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "bm25_score": 0.0,
                    "vector_score": chunk.score,
                }

        # Compute fusion scores
        results = []
        for chunk_id, data in merged.items():
            fusion_score = (
                BM25_WEIGHT * data["bm25_score"]
                + VECTOR_WEIGHT * data["vector_score"]
            )
            results.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    content=data["content"],
                    document_id=data["document_id"],
                    chunk_index=data["chunk_index"],
                    score=fusion_score,
                    source="hybrid",
                )
            )

        # Sort by score descending, return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
