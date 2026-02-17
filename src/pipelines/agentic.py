"""
Agentic RAG Pipeline for MedQuery

ReAct-style agent loop that classifies queries, decomposes complex ones
into sub-queries, performs iterative retrieval, and self-reflects on
answer completeness before returning.
"""

import logging
import re
import time
import uuid
from typing import Any

from src.pipelines.classical import PipelineResult
from src.pipelines.prompts import (
    CLASSIFY_PROMPT,
    DECOMPOSE_COUNTS,
    DECOMPOSE_PROMPT,
    MAX_SUB_QUERIES,
    REFLECT_PROMPT,
    VALID_QUERY_TYPES,
    build_synthesis_prompt,
)

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3


class AgenticPipeline:
    """ReAct-style agentic RAG pipeline with classify/decompose/reflect."""

    def __init__(self, embedding_generator, ollama_client, reranker):
        self.embedding_generator = embedding_generator
        self.ollama_client = ollama_client
        self.reranker = reranker

    async def run(
        self,
        question: str,
        max_results: int = 3,
        confidence_threshold: float = 0.70,
    ) -> PipelineResult:
        """Execute the agentic RAG pipeline."""
        start_time = time.time()
        steps: list[dict[str, Any]] = []

        # --- PII redaction on input ---
        step_start = time.time()
        from src.security.pii_redaction import PIIRedactor

        redactor = PIIRedactor()
        question = redactor.redact(question)
        steps.append(
            {
                "name": "pii_redact_input",
                "duration_ms": round((time.time() - step_start) * 1000, 1),
                "detail": "Redacted PII from query",
            }
        )

        # --- Check prerequisites ---
        if not self.embedding_generator:
            elapsed = (time.time() - start_time) * 1000
            return PipelineResult(
                answer="Embedding model not available. Please try again later.",
                confidence=0.0,
                citations=[],
                retrieved_chunks=[],
                query_id="error",
                processing_time_ms=round(elapsed, 1),
                pipeline="agentic",
                steps=steps,
            )

        # --- Step 1: Classify ---
        step_start = time.time()
        query_type = await self._classify(question)
        steps.append(
            {
                "name": "classify",
                "duration_ms": round((time.time() - step_start) * 1000, 1),
                "detail": query_type,
            }
        )

        # --- Step 2: Decompose ---
        step_start = time.time()
        if query_type == "SIMPLE":
            sub_queries = [question]
        else:
            sub_queries = await self._decompose(question, query_type)
        steps.append(
            {
                "name": "decompose",
                "duration_ms": round((time.time() - step_start) * 1000, 1),
                "detail": sub_queries,
            }
        )

        # --- Step 3: Iterative retrieval per sub-query ---
        from sqlalchemy import select

        from src.db.models import DocumentChunk
        from src.db.postgres import AsyncSessionLocal
        from src.rag.retriever import HybridRetriever, ScoredChunk

        all_chunks: list[ScoredChunk] = []
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(DocumentChunk))
                db_chunks = result.scalars().all()

                if not db_chunks:
                    elapsed = (time.time() - start_time) * 1000
                    return PipelineResult(
                        answer="No documents indexed yet. Upload documents first.",
                        confidence=0.0,
                        citations=[],
                        retrieved_chunks=[],
                        query_id="no_docs",
                        processing_time_ms=round(elapsed, 1),
                        pipeline="agentic",
                        steps=steps,
                    )

                for i, sq in enumerate(sub_queries):
                    step_start = time.time()
                    try:
                        embeddings = await self.embedding_generator.generate_embeddings(
                            [sq], is_query=True
                        )
                        query_embedding = embeddings[0]

                        retriever = HybridRetriever(db_chunks, session)
                        search_results = await retriever.search(
                            sq, query_embedding, top_k=max_results
                        )

                        # Rerank
                        if self.reranker and search_results:
                            try:
                                reranked = await self.reranker.rerank(
                                    sq, search_results, top_k=max_results
                                )
                                search_results = [
                                    ScoredChunk(
                                        chunk_id=r.chunk_id,
                                        content=r.content,
                                        document_id=r.document_id,
                                        chunk_index=r.chunk_index,
                                        score=r.final_score,
                                        source=r.source,
                                    )
                                    for r in reranked
                                ]
                            except Exception as e:
                                logger.warning("Reranking failed for sub-query: %s", e)

                        all_chunks.extend(search_results)
                    except Exception as e:
                        logger.warning("Retrieval failed for sub-query '%s': %s", sq, e)

                    steps.append(
                        {
                            "name": "retrieve",
                            "duration_ms": round((time.time() - step_start) * 1000, 1),
                            "detail": f"Sub-query {i + 1}: {sq}",
                        }
                    )
        except Exception as e:
            logger.warning("Database query failed: %s", e)
            elapsed = (time.time() - start_time) * 1000
            return PipelineResult(
                answer="Database unavailable. Please try again later.",
                confidence=0.0,
                citations=[],
                retrieved_chunks=[],
                query_id="error",
                processing_time_ms=round(elapsed, 1),
                pipeline="agentic",
                steps=steps,
            )

        # --- Step 4: Deduplicate combined pool ---
        step_start = time.time()
        unique_chunks = self._deduplicate(all_chunks)
        final_chunks = unique_chunks[: max_results * 2]
        steps.append(
            {
                "name": "deduplicate",
                "duration_ms": round((time.time() - step_start) * 1000, 1),
                "detail": (
                    f"{len(all_chunks)} total -> {len(unique_chunks)} unique "
                    f"-> {len(final_chunks)} kept"
                ),
            }
        )

        if not final_chunks:
            elapsed = (time.time() - start_time) * 1000
            return PipelineResult(
                answer="No relevant results found for your query.",
                confidence=0.0,
                citations=[],
                retrieved_chunks=[],
                query_id=str(uuid.uuid4())[:8],
                processing_time_ms=round(elapsed, 1),
                pipeline="agentic",
                steps=steps,
            )

        # --- Format chunks for synthesis ---
        retrieved_chunks_dicts: list[dict[str, Any]] = []
        for r in final_chunks:
            retrieved_chunks_dicts.append(
                {
                    "chunk_id": r.chunk_id,
                    "text": r.content,
                    "document_id": r.document_id,
                    "chunk_index": r.chunk_index,
                    "relevance_score": round(r.score, 4),
                    "source": r.source,
                }
            )

        # --- Step 5: Synthesize with agent-aware prompt ---
        step_start = time.time()
        answer, confidence, citations, hallucination_flagged = await self._synthesize(
            question, final_chunks, query_type, confidence_threshold, redactor
        )
        steps.append(
            {
                "name": "synthesize",
                "duration_ms": round((time.time() - step_start) * 1000, 1),
                "detail": "Generating answer with agent-aware prompt",
            }
        )

        # --- Step 6: Self-reflect ---
        if confidence < confidence_threshold:
            step_start = time.time()
            reflection = await self._reflect(question, answer, query_type, confidence)
            steps.append(
                {
                    "name": "reflect",
                    "duration_ms": round((time.time() - step_start) * 1000, 1),
                    "detail": reflection,
                }
            )

            # If reflection says insufficient, do one more retrieval pass
            if (
                reflection.startswith("INSUFFICIENT")
                and len(sub_queries) < MAX_ITERATIONS
            ):
                refined_query = self._extract_refined_query(reflection, question)
                step_start = time.time()
                try:
                    async with AsyncSessionLocal() as session:
                        result = await session.execute(select(DocumentChunk))
                        db_chunks = result.scalars().all()
                        if db_chunks:
                            embeddings = (
                                await self.embedding_generator.generate_embeddings(
                                    [refined_query], is_query=True
                                )
                            )
                            retriever = HybridRetriever(db_chunks, session)
                            extra = await retriever.search(
                                refined_query, embeddings[0], top_k=max_results
                            )
                            if self.reranker and extra:
                                try:
                                    reranked = await self.reranker.rerank(
                                        refined_query, extra, top_k=max_results
                                    )
                                    extra = [
                                        ScoredChunk(
                                            chunk_id=r.chunk_id,
                                            content=r.content,
                                            document_id=r.document_id,
                                            chunk_index=r.chunk_index,
                                            score=r.final_score,
                                            source=r.source,
                                        )
                                        for r in reranked
                                    ]
                                except Exception:
                                    pass

                            # Merge with existing
                            combined = self._deduplicate(final_chunks + extra)
                            final_chunks = combined[: max_results * 2]

                            # Re-synthesize
                            (
                                answer,
                                confidence,
                                citations,
                                hallucination_flagged,
                            ) = await self._synthesize(
                                question,
                                final_chunks,
                                query_type,
                                confidence_threshold,
                                redactor,
                            )

                            # Update retrieved_chunks_dicts
                            retrieved_chunks_dicts = [
                                {
                                    "chunk_id": r.chunk_id,
                                    "text": r.content,
                                    "document_id": r.document_id,
                                    "chunk_index": r.chunk_index,
                                    "relevance_score": round(r.score, 4),
                                    "source": r.source,
                                }
                                for r in final_chunks
                            ]
                except Exception as e:
                    logger.warning("Reflection retrieval failed: %s", e)

                steps.append(
                    {
                        "name": "retry_retrieve",
                        "duration_ms": round((time.time() - step_start) * 1000, 1),
                        "detail": f"Refined query: {refined_query}",
                    }
                )
        else:
            steps.append(
                {
                    "name": "reflect",
                    "duration_ms": 0,
                    "detail": "SUFFICIENT",
                }
            )

        # --- Complete ---
        steps.append({"name": "complete", "duration_ms": 0, "detail": "Done"})

        elapsed = (time.time() - start_time) * 1000
        return PipelineResult(
            answer=answer,
            confidence=confidence,
            citations=citations,
            retrieved_chunks=retrieved_chunks_dicts,
            query_id=str(uuid.uuid4())[:8],
            processing_time_ms=round(elapsed, 1),
            hallucination_flagged=hallucination_flagged,
            pipeline="agentic",
            steps=steps,
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    async def _classify(self, question: str) -> str:
        """Classify the query type using the LLM."""
        if not self.ollama_client:
            return "SIMPLE"

        try:
            prompt = CLASSIFY_PROMPT.format(question=question)
            raw = await self.ollama_client.generate(prompt)
            category = raw.strip().upper().split("\n")[0].strip()
            # Strip any trailing punctuation or extra words
            for valid in VALID_QUERY_TYPES:
                if valid in category:
                    return valid
            return "SIMPLE"
        except Exception as e:
            logger.warning("Classification failed, defaulting to SIMPLE: %s", e)
            return "SIMPLE"

    async def _decompose(self, question: str, query_type: str) -> list[str]:
        """Decompose a complex query into sub-queries."""
        if not self.ollama_client:
            return [question]

        n = DECOMPOSE_COUNTS.get(query_type, 2)
        try:
            prompt = DECOMPOSE_PROMPT.format(
                question=question, query_type=query_type, n=n
            )
            raw = await self.ollama_client.generate(prompt)
            sub_queries = self._parse_numbered_list(raw)

            if not sub_queries:
                return [question]

            return sub_queries[:MAX_SUB_QUERIES]
        except Exception as e:
            logger.warning("Decomposition failed, using original query: %s", e)
            return [question]

    async def _synthesize(
        self,
        question: str,
        chunks: list,
        query_type: str,
        confidence_threshold: float,
        redactor,
    ) -> tuple[str, float, list[dict[str, Any]], bool]:
        """Synthesize an answer from retrieved chunks.

        Returns (answer, confidence, citations, hallucination_flagged).
        """
        from src.llm.response_parser import ResponseParser

        # Build context string
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = getattr(chunk, "source", "Unknown")
            context_parts.append(f"[Source {i}: {source}]\n{chunk.content}")
        context = "\n\n".join(context_parts)

        if not self.ollama_client:
            confidence = chunks[0].score if chunks else 0.0
            return (
                f"Found {len(chunks)} relevant chunk(s). "
                "See retrieved_chunks for details. (LLM synthesis unavailable)",
                round(confidence, 4),
                [],
                False,
            )

        try:
            prompt = build_synthesis_prompt(question, context, query_type)
            raw_response = await self.ollama_client.generate(prompt)

            if raw_response:
                parser = ResponseParser()
                retrieved_for_validation = [
                    {"text": c.content, "source": getattr(c, "source", "")}
                    for c in chunks
                ]
                parsed = parser.parse(raw_response, retrieved_for_validation)

                if parsed.confidence < confidence_threshold:
                    answer = (
                        "Confidence too low for synthesis. "
                        "Relevant snippets are provided in retrieved_chunks."
                    )
                else:
                    answer = parsed.answer

                answer = redactor.redact(answer)

                citations = [
                    {
                        "document": c.document,
                        "section": c.section,
                        "page": c.page,
                        "relevance_score": round(chunks[0].score, 4) if chunks else 0.0,
                    }
                    for c in parsed.citations
                ]

                return (
                    answer,
                    round(parsed.confidence, 4),
                    citations,
                    parsed.has_hallucinated_citations,
                )
            else:
                logger.warning("Ollama returned empty, falling back to snippets")
        except Exception as e:
            logger.warning("Synthesis failed: %s", e)

        confidence = chunks[0].score if chunks else 0.0
        return (
            f"Found {len(chunks)} relevant chunk(s). "
            "See retrieved_chunks for details. (LLM synthesis unavailable)",
            round(confidence, 4),
            [],
            False,
        )

    async def _reflect(
        self, question: str, answer: str, query_type: str, confidence: float
    ) -> str:
        """Self-reflect on answer completeness."""
        if not self.ollama_client:
            return "SUFFICIENT"

        try:
            prompt = REFLECT_PROMPT.format(
                question=question,
                query_type=query_type,
                answer=answer,
                confidence=confidence,
            )
            raw = await self.ollama_client.generate(prompt)
            result = raw.strip()
            if result.upper().startswith("SUFFICIENT"):
                return "SUFFICIENT"
            if result.upper().startswith("INSUFFICIENT"):
                return result
            return "SUFFICIENT"
        except Exception as e:
            logger.warning("Reflection failed: %s", e)
            return "SUFFICIENT"

    @staticmethod
    def _extract_refined_query(reflection: str, fallback: str) -> str:
        """Extract the refined search query from a reflection response."""
        # Look for an explicit "Refined query:" line first
        rq_match = re.search(
            r"[Rr]efined\s+query[:\s]+[\"']?(.+?)[\"']?\s*$",
            reflection,
            re.MULTILINE,
        )
        if rq_match and rq_match.group(1).strip():
            return rq_match.group(1).strip()

        # Fall back to first non-empty line after the INSUFFICIENT line
        lines = reflection.strip().split("\n")
        for line in lines[1:]:
            stripped = line.strip()
            if stripped:
                # Strip leading "Refined query:" prefix if present
                cleaned = re.sub(r"^[Rr]efined\s+query[:\s]+", "", stripped)
                return cleaned.strip() or stripped
        # Try to extract from the INSUFFICIENT line itself
        match = re.search(r"INSUFFICIENT:\s*(.+)", reflection, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return fallback

    @staticmethod
    def _deduplicate(chunks: list) -> list:
        """Deduplicate chunks by chunk_id, keeping the highest-scored version."""
        seen: dict[int, Any] = {}
        for chunk in chunks:
            cid = chunk.chunk_id
            if cid not in seen or chunk.score > seen[cid].score:
                seen[cid] = chunk
        result = list(seen.values())
        result.sort(key=lambda x: x.score, reverse=True)
        return result

    @staticmethod
    def _parse_numbered_list(text: str) -> list[str]:
        """Parse a numbered list from LLM output."""
        lines = text.strip().split("\n")
        results = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip leading number + punctuation: "1. ", "1) ", "1: "
            cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", line)
            if cleaned:
                results.append(cleaned)
        return results
