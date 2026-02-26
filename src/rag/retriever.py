"""
Hybrid Retrieval System for MedQuery

Combines BM25 keyword search with pgvector cosine similarity search
using a weighted fusion strategy: 0.4 * BM25 + 0.6 * cosine.

Enhanced with:
- Multi-query retrieval: LLM generates query reformulations for broader recall
- Context window expansion: fetches adjacent chunks for richer context
- Entity-aware boosting: boosts chunks containing medical entities from the query
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
    "UTI": "urinary tract infection",
    "URI": "upper respiratory infection",
    "OPD": "outpatient department",
    "DOH": "department of health",
    "CPG": "clinical practice guidelines",
    "BP": "blood pressure",
    "HR": "heart rate",
    "RR": "respiratory rate",
    "ORS": "oral rehydration solution",
    "IV": "intravenous",
    "IM": "intramuscular",
    "PO": "per oral",
    "BID": "twice daily",
    "TID": "three times daily",
    "QID": "four times daily",
    "PRN": "as needed",
    "CBC": "complete blood count",
    "ABG": "arterial blood gas",
    "ECG": "electrocardiogram",
    "CXR": "chest x-ray",
}

# Natural-language synonyms → clinical terms (case-insensitive expansion)
# These are matched case-insensitively and appended to the query for broader recall
MEDICAL_SYNONYMS: dict[str, str] = {
    "heart attack": "myocardial infarction",
    "high blood pressure": "hypertension",
    "low blood pressure": "hypotension",
    "stroke": "cerebrovascular accident",
    "blood clot": "thrombosis embolism",
    "kidney failure": "renal failure",
    "liver failure": "hepatic failure",
    "sugar": "glucose diabetes",
    "blood sugar": "blood glucose diabetes",
    "bone break": "fracture",
    "broken bone": "fracture",
    "breathing difficulty": "dyspnea respiratory distress",
    "chest pain": "angina chest pain",
    "water pills": "diuretics",
    "blood thinner": "anticoagulant",
    "pain killer": "analgesic",
    "fever": "pyrexia febrile",
    "dengue": "dengue hemorrhagic fever",
    "leptospirosis": "leptospirosis leptospira",
    "pneumonia": "pneumonia community-acquired hospital-acquired",
    "TB": "tuberculosis",
    "diarrhea": "diarrhea acute gastroenteritis",
}

STOP_WORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "our",
    "out",
    "own",
    "she",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "too",
    "up",
    "us",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "would",
    "you",
    "your",
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
    document_name: str = ""  # Actual filename from clinical_docs


# ============================================
# QueryPreprocessor
# ============================================


# Pre-compiled abbreviation patterns (compiled once at module load, not per query)
_COMPILED_ABBREVIATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(abbr) + r"\b"), expansion)
    for abbr, expansion in MEDICAL_ABBREVIATIONS.items()
]


class QueryPreprocessor:
    """Preprocesses queries with normalization, abbreviation expansion, and tokenization."""

    def __init__(self):
        self._abbreviations = MEDICAL_ABBREVIATIONS
        self._synonyms = MEDICAL_SYNONYMS
        self._stop_words = STOP_WORDS

    def preprocess(self, query: str) -> str:
        """Lowercase, expand abbreviations, and expand medical synonyms."""
        result = self._expand_abbreviations(query)
        result = self._expand_synonyms(result)
        return result.lower()

    def _expand_abbreviations(self, text: str) -> str:
        """Replace medical abbreviations with full terms."""
        for pattern, expansion in _COMPILED_ABBREVIATION_PATTERNS:
            text = pattern.sub(expansion, text)
        return text

    def _expand_synonyms(self, text: str) -> str:
        """Append clinical synonyms for common natural-language terms."""
        text_lower = text.lower()
        additions = []
        for phrase, expansion in self._synonyms.items():
            if phrase in text_lower:
                additions.append(expansion)
        if additions:
            text = text + " " + " ".join(additions)
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
            # Format vector as pgvector literal: '[1.0,2.0,3.0]'::vector
            embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
            sql = text(
                "SELECT id, chunk_text, document_id, chunk_index, "
                "1 - (embedding <=> CAST(:query_vector AS vector)) AS similarity "
                "FROM embeddings_cache "
                "WHERE embedding IS NOT NULL "
                "ORDER BY embedding <=> CAST(:query_vector AS vector) "
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
        except Exception as e:
            logger.error(
                "Vector search failed: %s (query_vector dim=%d, top_k=%d)",
                str(e),
                len(query_embedding),
                top_k,
                exc_info=True,
            )
            return []


# ============================================
# HybridRetriever
# ============================================

BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6


class HybridRetriever:
    """Combines BM25 and vector search with weighted fusion scoring."""

    def __init__(
        self,
        chunks: list,
        session: AsyncSession,
        doc_name_map: dict[int, str] | None = None,
    ):
        self._bm25 = BM25Retriever(chunks)
        self._vector = VectorRetriever(session)
        self._doc_name_map = doc_name_map or {}

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
                BM25_WEIGHT * data["bm25_score"] + VECTOR_WEIGHT * data["vector_score"]
            )
            doc_id = data["document_id"]
            results.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    content=data["content"],
                    document_id=doc_id,
                    chunk_index=data["chunk_index"],
                    score=fusion_score,
                    source="hybrid",
                    document_name=self._doc_name_map.get(doc_id, ""),
                )
            )

        # Sort by score descending, return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


# ============================================
# Multi-Query Retrieval
# ============================================

MULTI_QUERY_PROMPT = """Rephrase this medical query {n} ways using different terminology. One per line, numbered 1-{n}. No other text.

Query: {query}"""


async def generate_query_variants(
    query: str,
    ollama_client,
    n: int = 3,
) -> list[str]:
    """Generate query reformulations via LLM for multi-query retrieval.

    Returns the original query plus up to n variants. Falls back to
    just the original query if LLM is unavailable.
    """
    if not ollama_client:
        return [query]

    try:
        prompt = MULTI_QUERY_PROMPT.format(query=query, n=n)
        raw = await ollama_client.generate(prompt)
        if not raw:
            return [query]

        variants = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", line)
            if cleaned and cleaned.lower() != query.lower():
                variants.append(cleaned)

        # Return original + variants (deduplicated)
        result = [query] + variants[:n]
        return result
    except Exception as e:
        logger.warning("Multi-query generation failed: %s", e)
        return [query]


# ============================================
# Context Window Expansion
# ============================================


async def expand_context_window(
    chunks: list[ScoredChunk],
    session: AsyncSession,
    window: int = 1,
    doc_name_map: dict[int, str] | None = None,
) -> list[ScoredChunk]:
    """Fetch adjacent chunks (chunk_index ± window) for each retrieved chunk.

    Expands context by fetching neighboring chunks from the same document,
    giving the LLM more surrounding context for synthesis.
    """
    if not chunks:
        return chunks

    # Collect (document_id, chunk_index) pairs to fetch
    existing_ids = {c.chunk_id for c in chunks}
    fetch_pairs: list[tuple[int, int]] = []
    for chunk in chunks:
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            neighbor_idx = chunk.chunk_index + offset
            if neighbor_idx >= 0:
                fetch_pairs.append((chunk.document_id, neighbor_idx))

    if not fetch_pairs:
        return chunks

    # Batch-fetch adjacent chunks
    try:
        # Build WHERE clause for all pairs
        conditions = " OR ".join(
            f"(document_id = {doc_id} AND chunk_index = {idx})"
            for doc_id, idx in fetch_pairs
        )
        sql = text(
            f"SELECT id, chunk_text, document_id, chunk_index "
            f"FROM embeddings_cache "
            f"WHERE ({conditions}) AND id NOT IN ({','.join(str(i) for i in existing_ids)})"
        )
        result = await session.execute(sql)
        rows = result.fetchall()

        # Add adjacent chunks with a reduced score
        _name_map = doc_name_map or {}
        expanded = list(chunks)
        for row in rows:
            if row.id not in existing_ids:
                # Adjacent chunks get a fraction of the nearest retrieved chunk's score
                parent_score = next(
                    (c.score for c in chunks if c.document_id == row.document_id),
                    0.3,
                )
                expanded.append(
                    ScoredChunk(
                        chunk_id=row.id,
                        content=row.chunk_text,
                        document_id=row.document_id,
                        chunk_index=row.chunk_index,
                        score=parent_score * 0.5,  # Adjacent = half the parent score
                        source="context_expansion",
                        document_name=_name_map.get(row.document_id, ""),
                    )
                )
                existing_ids.add(row.id)

        return expanded
    except Exception as e:
        logger.warning("Context window expansion failed: %s", e)
        return chunks


# ============================================
# Entity-Aware Retrieval Boost
# ============================================

# Medical entity patterns for extraction (drug names, conditions, procedures)
MEDICAL_ENTITY_PATTERNS = [
    # Drug-like terms (capitalized multi-word or specific suffixes)
    r"\b[A-Z][a-z]+(?:mycin|cillin|olol|pril|sartan|statin|azole|prazole|mab|nib|tinib)\b",
    # Dosage patterns
    r"\b\d+\s*(?:mg|mcg|ml|g|units?|IU)\b",
    # Common procedure terms
    r"\b(?:biopsy|endoscopy|colonoscopy|MRI|CT scan|X-ray|ultrasound|echocardiogram|catheterization|angiography|transplant)\b",
]


def extract_medical_entities(query: str) -> list[str]:
    """Extract medical entities from a query string.

    Returns a list of medical terms found (drug names, conditions,
    procedures, dosages) using pattern matching + abbreviation dictionary.
    """
    entities = []

    # Check for known medical abbreviations (using pre-compiled patterns)
    for pattern, expansion in _COMPILED_ABBREVIATION_PATTERNS:
        match = pattern.search(query)
        if match:
            entities.append(match.group(0).lower())
            entities.append(expansion.lower())

    # Check for medical entity patterns
    for pattern in MEDICAL_ENTITY_PATTERNS:
        for match in re.finditer(pattern, query, re.IGNORECASE):
            entities.append(match.group(0).lower())

    # Extract capitalized multi-word terms (likely medical terms)
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query):
        term = match.group(0).lower()
        if term not in STOP_WORDS and len(term) > 3:
            entities.append(term)

    return list(set(entities))


def boost_entity_matches(
    chunks: list[ScoredChunk],
    entities: list[str],
    boost_factor: float = 1.15,
) -> list[ScoredChunk]:
    """Boost scores of chunks containing medical entities from the query.

    Chunks matching more entities get progressively higher boosts.
    """
    if not entities or not chunks:
        return chunks

    boosted = []
    for chunk in chunks:
        content_lower = chunk.content.lower()
        matches = sum(1 for entity in entities if entity in content_lower)
        if matches > 0:
            # Progressive boost: each entity match adds boost_factor multiplier
            multiplier = 1.0 + (boost_factor - 1.0) * min(matches, 3)
            boosted.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    score=min(chunk.score * multiplier, 1.0),
                    source=chunk.source,
                    document_name=chunk.document_name,
                )
            )
        else:
            boosted.append(chunk)

    boosted.sort(key=lambda x: x.score, reverse=True)
    return boosted
