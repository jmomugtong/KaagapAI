"""
Response Parser for KaagapAI

Parses LLM output to extract:
- Answer text (without metadata lines)
- Confidence score (0.0-1.0)
- Citations in [Document Name, Section, p. Page] format
- Hallucination detection (citations not in retrieval set)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Regex for confidence extraction: matches "Confidence: 0.92", "Confidence Score: 0.85",
# "**Confidence:** 0.85", "Confidence Level: 0.80", parenthesized variants
CONFIDENCE_PATTERN = re.compile(
    r"\*{0,2}[Cc]onfidence(?:\s+(?:[Ss]core|[Ll]evel))?\*{0,2}[:\s]+([\d.]+)",
    re.IGNORECASE,
)

# Regex for bracket citations: [Document Name, Section X, p. Y]
CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")

DEFAULT_CONFIDENCE = 0.50


@dataclass
class Citation:
    """A parsed citation from LLM output."""

    document: str
    section: str | None = None
    page: int | None = None


@dataclass
class ParsedResponse:
    """Structured output from parsing an LLM response."""

    answer: str
    confidence: float
    citations: list[Citation] = field(default_factory=list)
    has_hallucinated_citations: bool = False


class ResponseParser:
    """Parses raw LLM text into structured response with citations."""

    def parse(
        self,
        raw_text: str,
        retrieved_chunks: list[dict[str, Any]],
        fallback_confidence: float | None = None,
    ) -> ParsedResponse:
        """
        Parse raw LLM output into a structured response.

        Args:
            raw_text: Raw text from the LLM.
            retrieved_chunks: List of dicts with 'source' key for validation.
            fallback_confidence: Confidence to use when the LLM doesn't output one.
                Typically the top retrieval score. Falls back to DEFAULT_CONFIDENCE
                if not provided.

        Returns:
            ParsedResponse with answer, confidence, citations, hallucination flag.
        """
        if not raw_text:
            return ParsedResponse(
                answer="",
                confidence=fallback_confidence or DEFAULT_CONFIDENCE,
                citations=[],
                has_hallucinated_citations=False,
            )

        confidence = self._extract_confidence(raw_text)
        # If LLM didn't output confidence, use retrieval-based fallback
        if confidence == DEFAULT_CONFIDENCE and fallback_confidence is not None:
            confidence = fallback_confidence
        answer = self._extract_answer(raw_text)
        citations = self._extract_citations(raw_text)
        has_hallucinated = self._check_hallucinations(citations, retrieved_chunks)

        return ParsedResponse(
            answer=answer,
            confidence=confidence,
            citations=citations,
            has_hallucinated_citations=has_hallucinated,
        )

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from text, clamped to [0.0, 1.0]."""
        match = CONFIDENCE_PATTERN.search(text)
        if match:
            try:
                value = float(match.group(1))
                return max(0.0, min(1.0, value))
            except ValueError:
                pass
        return DEFAULT_CONFIDENCE

    def _extract_answer(self, text: str) -> str:
        """Extract the answer text, stripping confidence metadata."""
        # Remove confidence pattern wherever it appears (end of line or standalone)
        text = CONFIDENCE_PATTERN.sub("", text)
        lines = text.strip().split("\n")
        answer_lines = [line for line in lines if line.strip()]
        return "\n".join(answer_lines).strip()

    def _extract_citations(self, text: str) -> list[Citation]:
        """Extract citations in [Document, Section, p. Page] format."""
        citations = []
        for match in CITATION_PATTERN.finditer(text):
            inner = match.group(1).strip()

            # Skip things that are clearly not citations
            # (e.g., "[Source 1: ...]" from prompt context is not a citation)
            if inner.startswith("Source ") and ":" in inner:
                continue

            # Skip single-word brackets like [1], [Note], [a] — not real citations
            if "," not in inner and len(inner) <= 10:
                continue

            parts = [p.strip() for p in inner.split(",")]
            if not parts:
                continue

            document = parts[0]
            section = None
            page = None

            for part in parts[1:]:
                # Check for page reference
                page_match = re.match(r"p\.?\s*(\d+)", part, re.IGNORECASE)
                if page_match:
                    page = int(page_match.group(1))
                elif section is None:
                    section = part

            citations.append(Citation(document=document, section=section, page=page))

        return citations

    def _check_hallucinations(
        self,
        citations: list[Citation],
        retrieved_chunks: list[dict[str, Any]],
    ) -> bool:
        """
        Check if any citation references a document not in the retrieval set.

        Returns True if hallucinated citations are detected.
        """
        if not citations:
            return False

        # Collect all source names from retrieved chunks
        known_sources: set[str] = set()
        for chunk in retrieved_chunks:
            source = chunk.get("source", "")
            if source:
                known_sources.add(source.lower())

        # If no sources to validate against, can't detect hallucination
        if not known_sources:
            return False

        # Check each citation against known sources using token overlap
        for citation in citations:
            # "Source N" references are from our own context headers — always valid
            if re.match(r"^Source\s+\d+$", citation.document, re.IGNORECASE):
                continue

            # "Document Name" is the placeholder from the prompt template — not hallucinated
            if citation.document.lower() == "document name":
                continue

            citation_tokens = self._tokenize_name(citation.document)
            if not citation_tokens:
                continue
            matched = any(
                self._token_overlap(citation_tokens, self._tokenize_name(known)) >= 0.5
                for known in known_sources
            )
            if not matched:
                logger.warning(
                    "Hallucinated citation detected: '%s' not in retrieval set %s",
                    citation.document,
                    known_sources,
                )
                return True

        return False

    @staticmethod
    def _tokenize_name(name: str) -> set[str]:
        """Tokenize a document name for fuzzy matching.

        Splits on delimiters (_-./\\, space), lowercases, removes file
        extensions and very short tokens.
        """
        # Remove common file extensions
        name = re.sub(r"\.(pdf|docx?|txt|csv|xlsx?)$", "", name, flags=re.IGNORECASE)
        tokens = re.split(r"[_\-./\\\s,]+", name.lower())
        return {t for t in tokens if len(t) > 1}

    @staticmethod
    def _token_overlap(tokens_a: set[str], tokens_b: set[str]) -> float:
        """Compute fraction of tokens_a that appear in tokens_b."""
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a)
