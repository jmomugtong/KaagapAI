"""
Prompt Templates for KaagapAI

Clinical QA prompt template that injects retrieved context chunks
and instructs the LLM to produce cited, confidence-scored answers.
"""

import os
import re
from typing import Any

# GRADE evidence-assessment vocabulary (case-insensitive match)
_GRADE_WORDS = frozenset({
    "serious", "not serious", "very serious",
    "low", "very low", "moderate", "high", "critical",
    "undetected", "not detected", "none",
    "important", "not important",
})


def _is_table_line(line: str) -> bool:
    """Return True if a line looks like tabular noise rather than prose."""
    stripped = line.strip()
    if not stripped:
        return False

    # Pipe-delimited table rows (2+ pipes)
    if stripped.count("|") >= 2:
        return True

    # Whitespace-aligned columns: 2+ gaps of 3+ consecutive spaces
    if len(re.findall(r" {3,}", stripped)) >= 2:
        return True

    # GRADE vocabulary lines: short lines dominated by GRADE terms
    words = stripped.split()
    if 1 <= len(words) <= 8:
        lower = stripped.lower()
        if any(g in lower for g in _GRADE_WORDS):
            return True

    # Numeric-heavy sparse lines: >60% digits/parens with <5 words
    if len(words) < 5:
        digit_paren_chars = sum(1 for c in stripped if c.isdigit() or c in "(),.")
        if len(stripped) > 0 and digit_paren_chars / len(stripped) > 0.6:
            return True

    return False


def clean_chunk_text(text: str) -> str:
    """Remove tabular noise (GRADE tables, pipe-delimited rows, etc.).

    Safety: if ALL lines would be removed, return the original unchanged.
    """
    lines = text.split("\n")
    kept = [line for line in lines if not _is_table_line(line)]
    if not kept or all(not line.strip() for line in kept):
        return text
    return "\n".join(kept)


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at a sentence boundary, falling back to hard cut.

    Finds the last sentence-ending punctuation (. ? !) followed by a space
    or newline before the limit, but only if it's past the halfway point.
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    search_region = text[half:max_chars]
    # Find last sentence boundary in the search region
    match = None
    for m in re.finditer(r"[.!?](?:\s|\n)", search_region):
        match = m
    if match:
        cut = half + match.end()
        return text[:cut].rstrip()
    return text[:max_chars] + "..."


class PromptTemplate:
    """Builds prompts for clinical question answering."""

    TEMPLATE = """Answer using ONLY the context below. Include dosages, schedules, and criteria only if present in context. Do not add information beyond what the context provides. Be concise. Cite sources as [Document Name, Section]. If insufficient, say so.
Ignore any evidence quality rating tables (e.g., GRADE assessments with terms like Serious/Undetected/Very Low).

CONTEXT:
{context}

QUESTION: {question}

ANSWER (then on a new line write "Confidence: " followed by a score from 0.0 to 1.0):
"""

    def __init__(
        self,
        max_chunks: int | None = None,
        max_chunk_chars: int | None = None,
    ) -> None:
        self.max_chunks = max_chunks or int(
            os.environ.get("LLM_MAX_CONTEXT_CHUNKS", "3")
        )
        self.max_chunk_chars = max_chunk_chars or int(
            os.environ.get("LLM_MAX_CHUNK_CHARS", "400")
        )

    def build(
        self,
        question: str,
        chunks: list[dict[str, Any]],
    ) -> str:
        """
        Build a prompt from a question and retrieved context chunks.

        Args:
            question: The user's clinical query.
            chunks: List of dicts with 'text' and 'metadata' keys.
                    Only the top `max_chunks` are included.

        Returns:
            Formatted prompt string ready for LLM.
        """
        limited_chunks = chunks[: self.max_chunks]
        context = self._format_context(limited_chunks)
        return self.TEMPLATE.format(context=context, question=question)

    def _format_context(self, chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a numbered context block."""
        if not chunks:
            return "No context documents available."

        sections = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            text = clean_chunk_text(text)
            if len(text) > self.max_chunk_chars:
                text = truncate_at_sentence(text, self.max_chunk_chars)
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "Unknown Source")
            page = metadata.get("page")
            chunk_index = metadata.get("chunk_index")
            relevance = metadata.get("relevance_score")

            header = f"[Source {i}: {source}"
            if page is not None:
                header += f", p. {page}"
            if chunk_index is not None:
                header += f", chunk {chunk_index}"
            if relevance is not None:
                header += f", relevance {relevance:.0%}"
            header += "]"

            sections.append(f"{header}\n{text}")

        return "\n\n".join(sections)
