"""
Prompt Templates for MedQuery

Clinical QA prompt template that injects retrieved context chunks
and instructs the LLM to produce cited, confidence-scored answers.
"""

from typing import Any


class PromptTemplate:
    """Builds prompts for clinical question answering."""

    TEMPLATE = """You are a medical information assistant. Given the following context from clinical documents, answer the user's question accurately and cite your sources.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer concisely and accurately based ONLY on the provided context
2. Cite sources using [Document Name, Section, p. Page] format
3. If uncertain, state your limitations clearly
4. Do not add information not present in the context
5. Assign a confidence score (0.0-1.0) on the last line as "Confidence: X.XX"

RESPONSE:
"""

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

        Returns:
            Formatted prompt string ready for LLM.
        """
        context = self._format_context(chunks)
        return self.TEMPLATE.format(context=context, question=question)

    def _format_context(self, chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a numbered context block."""
        if not chunks:
            return "No context documents available."

        sections = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
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
