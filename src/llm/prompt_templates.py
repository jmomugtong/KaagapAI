"""
Prompt Templates for MedQuery

Clinical QA prompt template that injects retrieved context chunks
and instructs the LLM to produce cited, confidence-scored answers.
"""

import os
from typing import Any


class PromptTemplate:
    """Builds prompts for clinical question answering."""

    TEMPLATE = """You are a medical information assistant with strict grounding rules. Given ONLY the context below, answer the user's question. You MUST NOT use any knowledge outside the provided context.

CONTEXT:
{context}

QUESTION:
{question}

STRICT GROUNDING RULES:
1. Answer ONLY using information explicitly stated in the CONTEXT above
2. Quote or closely paraphrase exact phrases from the sources — do not rephrase medical facts in your own words
3. Cite every factual claim using [Document Name, Section, p. Page] format
4. If the context does NOT contain enough information to answer, respond: "The indexed documents do not contain sufficient information to answer this question."
5. Do NOT add medical advice, dosages, drug interactions, or treatment recommendations not explicitly present in the context
6. If multiple sources provide conflicting information, note the discrepancy
7. Assign a confidence score (0.0-1.0) based on how well the context supports your answer:
   - 0.90-1.0: Direct, explicit answer found in context
   - 0.70-0.89: Answer partially supported, some inference needed
   - Below 0.70: Weak support, mostly inference
8. Write the confidence on the last line as "Confidence: X.XX"

RESPONSE:
"""

    def __init__(
        self,
        max_chunks: int | None = None,
        max_chunk_chars: int | None = None,
    ) -> None:
        self.max_chunks = max_chunks or int(
            os.environ.get("LLM_MAX_CONTEXT_CHUNKS", "5")
        )
        self.max_chunk_chars = max_chunk_chars or int(
            os.environ.get("LLM_MAX_CHUNK_CHARS", "800")
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
            if len(text) > self.max_chunk_chars:
                text = text[: self.max_chunk_chars] + "..."
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "Unknown Source")
            page = metadata.get("page")
            chunk_index = metadata.get("chunk_index")

            header = f"[Source {i}: {source}"
            if page is not None:
                header += f", p. {page}"
            if chunk_index is not None:
                header += f", chunk {chunk_index}"
            header += "]"

            sections.append(f"{header}\n{text}")

        return "\n\n".join(sections)
