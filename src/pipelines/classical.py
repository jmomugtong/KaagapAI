"""
Classical RAG Pipeline for KaagapAI

Encapsulates the standard retrieve-rerank-synthesize flow as a callable unit.
Extracted from src/main.py to allow side-by-side comparison with the agentic pipeline.

Enhanced with:
- Multi-query retrieval: generates query reformulations for broader recall
- Context window expansion: fetches adjacent chunks for richer context
- Entity-aware boosting: boosts chunks containing medical entities
- Sentence-level extraction: extracts key sentences for focused context
- Extractive fallback: returns real document sentences when LLM confidence is low
- Web search fallback: searches the web when no local documents match
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Standardized result from any RAG pipeline."""

    answer: str
    confidence: float
    citations: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    query_id: str
    processing_time_ms: float
    hallucination_flagged: bool = False
    cached: bool = False
    pipeline: str = "classical"
    steps: list[dict[str, Any]] = field(default_factory=list)


class ClassicalPipeline:
    """Standard retrieve-rerank-synthesize RAG pipeline."""

    def __init__(
        self,
        embedding_generator,
        ollama_client,
        reranker,
        cached_chunks=None,
        doc_name_map=None,
    ):
        self.embedding_generator = embedding_generator
        self.ollama_client = ollama_client
        self.reranker = reranker
        self.cached_chunks = cached_chunks  # Pre-loaded chunks to avoid DB queries
        self.doc_name_map = doc_name_map or {}

    async def run(
        self,
        question: str,
        max_results: int = 3,
        confidence_threshold: float = 0.70,
    ) -> PipelineResult:
        """Execute the full classical RAG pipeline.

        Stages: validate inputs -> check cache -> embed -> retrieve ->
                rerank -> synthesize -> redact output -> cache result.
        """
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

        # --- Cache check ---
        step_start = time.time()
        from src.rag.cache import CacheManager

        cache = CacheManager()
        try:
            cached = await cache.get_query_result(question)
            if cached is not None:
                elapsed = (time.time() - start_time) * 1000
                steps.append(
                    {
                        "name": "cache_hit",
                        "duration_ms": round((time.time() - step_start) * 1000, 1),
                        "detail": "Cache hit — returning cached result",
                    }
                )
                return PipelineResult(
                    answer=cached.get("answer", ""),
                    confidence=cached.get("confidence", 0.0),
                    citations=cached.get("citations", []),
                    retrieved_chunks=cached.get("retrieved_chunks", []),
                    query_id=cached.get("query_id", "cached"),
                    processing_time_ms=round(elapsed, 1),
                    hallucination_flagged=cached.get("hallucination_flagged", False),
                    cached=True,
                    pipeline="classical",
                    steps=steps,
                )
        except Exception as e:
            logger.warning("Query cache lookup failed: %s", e)
        steps.append(
            {
                "name": "cache_check",
                "duration_ms": round((time.time() - step_start) * 1000, 1),
                "detail": "Cache miss",
            }
        )

        # --- Embedding generation ---
        step_start = time.time()
        if not self.embedding_generator:
            elapsed = (time.time() - start_time) * 1000
            return PipelineResult(
                answer="Embedding model not available. Please try again later.",
                confidence=0.0,
                citations=[],
                retrieved_chunks=[],
                query_id="error",
                processing_time_ms=round(elapsed, 1),
                pipeline="classical",
                steps=steps,
            )

        try:
            embeddings = await self.embedding_generator.generate_embeddings(
                [question], is_query=True
            )
            _ = embeddings[0]  # Validate embedding generation works
        except Exception as e:
            logger.warning("Embedding generation failed: %s", e)
            elapsed = (time.time() - start_time) * 1000
            return PipelineResult(
                answer="Embedding generation failed. Please try again later.",
                confidence=0.0,
                citations=[],
                retrieved_chunks=[],
                query_id="error",
                processing_time_ms=round(elapsed, 1),
                pipeline="classical",
                steps=steps,
            )
        steps.append(
            {
                "name": "embed",
                "duration_ms": round((time.time() - step_start) * 1000, 1),
                "detail": "Generated query embedding",
            }
        )

        # --- Multi-query generation ---
        step_start = time.time()
        from src.rag.retriever import (
            HybridRetriever,
            ScoredChunk,
            boost_entity_matches,
            expand_context_window,
            extract_medical_entities,
            generate_query_variants,
        )

        use_multi_query = os.environ.get("MULTI_QUERY_ENABLED", "0") == "1"
        if use_multi_query:
            query_variants = await generate_query_variants(
                question, self.ollama_client, n=2
            )
        else:
            query_variants = [question]
        steps.append(
            {
                "name": "multi_query",
                "duration_ms": round((time.time() - step_start) * 1000, 1),
                "detail": f"Generated {len(query_variants)} query variants",
            }
        )

        # --- Hybrid retrieval (multi-query) ---
        step_start = time.time()
        from sqlalchemy import select

        from src.db.models import DocumentChunk
        from src.db.postgres import AsyncSessionLocal

        try:
            # Use cached chunks if available, otherwise load from DB
            if self.cached_chunks is not None:
                chunks = self.cached_chunks
                logger.info("Using cached chunks: %d chunks", len(chunks))
            else:
                logger.info("Loading chunks from database (cache not available)")
                async with AsyncSessionLocal() as session:
                    result = await session.execute(select(DocumentChunk))
                    chunks = result.scalars().all()

            if not chunks:
                elapsed = (time.time() - start_time) * 1000
                return PipelineResult(
                    answer="No documents indexed yet. Upload documents first.",
                    confidence=0.0,
                    citations=[],
                    retrieved_chunks=[],
                    query_id="no_docs",
                    processing_time_ms=round(elapsed, 1),
                    pipeline="classical",
                    steps=steps,
                )

            all_results: list[ScoredChunk] = []
            async with AsyncSessionLocal() as session:
                for variant in query_variants:
                    try:
                        variant_embeddings = (
                            await self.embedding_generator.generate_embeddings(
                                [variant], is_query=True
                            )
                        )
                        retriever = HybridRetriever(
                            chunks, session, doc_name_map=self.doc_name_map
                        )
                        variant_results = await retriever.search(
                            variant, variant_embeddings[0], top_k=max_results
                        )
                        all_results.extend(variant_results)
                    except Exception as e:
                        logger.warning(
                            "Retrieval failed for variant '%s': %s", variant, e
                        )

                # Deduplicate by chunk_id, keeping highest score
                seen: dict[int, ScoredChunk] = {}
                for chunk in all_results:
                    if (
                        chunk.chunk_id not in seen
                        or chunk.score > seen[chunk.chunk_id].score
                    ):
                        seen[chunk.chunk_id] = chunk
                search_results = sorted(
                    seen.values(), key=lambda x: x.score, reverse=True
                )

                # --- Entity-aware boosting ---
                entities = extract_medical_entities(question)
                if entities:
                    search_results = boost_entity_matches(search_results, entities)

                # --- Context window expansion (skipped by default for speed) ---
                if os.environ.get("SKIP_CONTEXT_EXPANSION", "1") != "1":
                    search_results = await expand_context_window(
                        search_results[:max_results],
                        session,
                        window=1,
                        doc_name_map=self.doc_name_map,
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
                pipeline="classical",
                steps=steps,
            )
        steps.append(
            {
                "name": "retrieve",
                "duration_ms": round((time.time() - step_start) * 1000, 1),
                "detail": (
                    f"Retrieved {len(search_results)} chunks via multi-query hybrid search "
                    f"({len(query_variants)} variants, {len(entities)} entities boosted)"
                ),
            }
        )

        # --- Reranking ---
        step_start = time.time()
        if self.reranker and search_results:
            try:
                reranked = await self.reranker.rerank(
                    question, search_results, top_k=max_results
                )
                search_results = [
                    ScoredChunk(
                        chunk_id=r.chunk_id,
                        content=r.content,
                        document_id=r.document_id,
                        chunk_index=r.chunk_index,
                        score=r.final_score,
                        source=r.source,
                        document_name=r.document_name,
                    )
                    for r in reranked
                ]
            except Exception as e:
                logger.warning("Reranking failed, using retrieval order: %s", e)
        steps.append(
            {
                "name": "rerank",
                "duration_ms": round((time.time() - step_start) * 1000, 1),
                "detail": f"Reranked to top {len(search_results)} chunks",
            }
        )

        # --- Format chunks ---
        min_relevance = float(os.environ.get("MIN_CHUNK_RELEVANCE", "0.35"))
        retrieved_chunks: list[dict[str, Any]] = []
        prompt_chunks: list[dict[str, Any]] = []
        for r in search_results:
            doc_display = r.document_name or f"Document {r.document_id}"
            retrieved_chunks.append(
                {
                    "chunk_id": r.chunk_id,
                    "text": r.content,
                    "document_id": r.document_id,
                    "chunk_index": r.chunk_index,
                    "relevance_score": round(r.score, 4),
                    "source": doc_display,
                }
            )
            # Only feed high-relevance chunks to the LLM to avoid noise
            if r.score >= min_relevance:
                prompt_chunks.append(
                    {
                        "text": r.content,
                        "metadata": {
                            "source": doc_display,
                            "chunk_index": r.chunk_index,
                            "document_id": r.document_id,
                            "relevance_score": round(r.score, 4),
                        },
                    }
                )

        if not search_results:
            # --- Web search fallback ---
            step_start = time.time()
            try:
                from src.rag.web_search import (
                    format_web_results_as_context,
                    search_web,
                    web_results_to_chunks,
                )

                web_results = await search_web(question, max_results=3)
                if web_results:
                    steps.append(
                        {
                            "name": "web_search_fallback",
                            "duration_ms": round((time.time() - step_start) * 1000, 1),
                            "detail": f"Found {len(web_results)} web results",
                        }
                    )
                    web_context = format_web_results_as_context(web_results)
                    web_chunk_dicts = web_results_to_chunks(web_results)

                    # Try LLM synthesis with web context
                    if self.ollama_client and web_context:
                        from src.llm.prompt_templates import PromptTemplate

                        template = PromptTemplate()
                        web_prompt_chunks = [
                            {"text": r.snippet, "metadata": {"source": r.title}}
                            for r in web_results
                        ]
                        prompt = template.build(
                            question=question, chunks=web_prompt_chunks
                        )
                        raw_response = await self.ollama_client.generate(prompt)
                        if raw_response:
                            from src.llm.response_parser import ResponseParser

                            parser = ResponseParser()
                            parsed = parser.parse(raw_response, [])
                            elapsed = (time.time() - start_time) * 1000
                            return PipelineResult(
                                answer=(
                                    "**Note: Answer based on web sources, not indexed clinical documents.**\n\n"
                                    + parsed.answer
                                ),
                                confidence=round(
                                    parsed.confidence * 0.7, 4
                                ),  # Discount web confidence
                                citations=[],
                                retrieved_chunks=web_chunk_dicts,
                                query_id=str(uuid.uuid4())[:8],
                                processing_time_ms=round(elapsed, 1),
                                pipeline="classical",
                                steps=steps,
                            )
            except Exception as e:
                logger.warning("Web search fallback failed: %s", e)

            elapsed = (time.time() - start_time) * 1000
            return PipelineResult(
                answer="No relevant results found for your query.",
                confidence=0.0,
                citations=[],
                retrieved_chunks=[],
                query_id=str(uuid.uuid4())[:8],
                processing_time_ms=round(elapsed, 1),
                pipeline="classical",
                steps=steps,
            )

        # --- LLM synthesis ---
        step_start = time.time()
        from src.llm.prompt_templates import PromptTemplate
        from src.llm.response_parser import ResponseParser

        if self.ollama_client:
            try:
                template = PromptTemplate()
                prompt = template.build(question=question, chunks=prompt_chunks)
                raw_response = await self.ollama_client.generate(prompt)

                if raw_response:
                    parser = ResponseParser()
                    top_score = search_results[0].score if search_results else None
                    parsed = parser.parse(
                        raw_response,
                        retrieved_chunks=[
                            {"text": c["text"], "source": c["source"]}
                            for c in retrieved_chunks
                        ],
                        fallback_confidence=top_score,
                    )

                    if parsed.confidence < confidence_threshold:
                        # Extractive fallback: return key sentences from docs
                        from src.rag.reranker import (
                            RerankedChunk,
                            build_extractive_answer,
                        )

                        extractive_chunks = [
                            RerankedChunk(
                                chunk_id=r.chunk_id,
                                content=r.content,
                                document_id=r.document_id,
                                chunk_index=r.chunk_index,
                                retrieval_score=r.score,
                                rerank_score=r.score,
                                final_score=r.score,
                                source=r.source,
                            )
                            for r in search_results
                        ]
                        answer = build_extractive_answer(extractive_chunks, question)
                    else:
                        answer = parsed.answer

                    answer = redactor.redact(answer)

                    citations = [
                        {
                            "document": c.document,
                            "section": c.section,
                            "page": c.page,
                            "relevance_score": round(search_results[0].score, 4),
                        }
                        for c in parsed.citations
                    ]

                    steps.append(
                        {
                            "name": "synthesize",
                            "duration_ms": round((time.time() - step_start) * 1000, 1),
                            "detail": "LLM synthesis complete",
                        }
                    )

                    elapsed = (time.time() - start_time) * 1000
                    response = PipelineResult(
                        answer=answer,
                        confidence=round(parsed.confidence, 4),
                        citations=citations,
                        retrieved_chunks=retrieved_chunks,
                        query_id=str(uuid.uuid4())[:8],
                        processing_time_ms=round(elapsed, 1),
                        hallucination_flagged=parsed.has_hallucinated_citations,
                        pipeline="classical",
                        steps=steps,
                    )

                    # Cache the response
                    try:
                        await cache.set_query_result(
                            question, self._to_cache_dict(response)
                        )
                    except Exception as e:
                        logger.warning("Failed to cache query result: %s", e)

                    return response
                else:
                    logger.warning(
                        "Ollama returned empty response, falling back to snippets"
                    )
            except Exception as e:
                logger.warning("LLM synthesis failed: %s, falling back to snippets", e)

        steps.append(
            {
                "name": "synthesize",
                "duration_ms": round((time.time() - step_start) * 1000, 1),
                "detail": "Fallback — snippets only (LLM unavailable)",
            }
        )

        # --- Fallback: snippets without LLM ---
        confidence = search_results[0].score if search_results else 0.0
        answer = (
            f"Found {len(search_results)} relevant chunk(s). "
            "See retrieved_chunks for details. (LLM synthesis unavailable)"
        )

        elapsed = (time.time() - start_time) * 1000
        return PipelineResult(
            answer=answer,
            confidence=round(confidence, 4),
            citations=[
                {
                    "document_id": r.document_id,
                    "chunk_index": r.chunk_index,
                    "relevance_score": round(r.score, 4),
                }
                for r in search_results
            ],
            retrieved_chunks=retrieved_chunks,
            query_id=str(uuid.uuid4())[:8],
            processing_time_ms=round(elapsed, 1),
            pipeline="classical",
            steps=steps,
        )

    @staticmethod
    def _to_cache_dict(result: PipelineResult) -> dict[str, Any]:
        """Convert PipelineResult to a dict suitable for caching."""
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "citations": result.citations,
            "retrieved_chunks": result.retrieved_chunks,
            "query_id": result.query_id,
            "processing_time_ms": result.processing_time_ms,
            "hallucination_flagged": result.hallucination_flagged,
        }
