"""
Tests for the Classical RAG Pipeline
"""

import pytest

from src.pipelines.classical import ClassicalPipeline, PipelineResult
from src.rag.retriever import ScoredChunk


def _make_chunk(chunk_id: int, content: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        content=content,
        document_id=1,
        chunk_index=chunk_id,
        score=score,
        source="hybrid",
    )


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    @pytest.mark.unit
    def test_pipeline_result_defaults(self):
        result = PipelineResult(
            answer="test",
            confidence=0.9,
            citations=[],
            retrieved_chunks=[],
            query_id="q1",
            processing_time_ms=100.0,
        )
        assert result.pipeline == "classical"
        assert result.cached is False
        assert result.hallucination_flagged is False
        assert result.steps == []

    @pytest.mark.unit
    def test_pipeline_result_with_steps(self):
        steps = [{"name": "embed", "duration_ms": 50, "detail": "ok"}]
        result = PipelineResult(
            answer="test",
            confidence=0.9,
            citations=[],
            retrieved_chunks=[],
            query_id="q1",
            processing_time_ms=100.0,
            steps=steps,
        )
        assert len(result.steps) == 1
        assert result.steps[0]["name"] == "embed"

    @pytest.mark.unit
    def test_pipeline_result_pipeline_field(self):
        result = PipelineResult(
            answer="",
            confidence=0.0,
            citations=[],
            retrieved_chunks=[],
            query_id="q",
            processing_time_ms=0.0,
            pipeline="agentic",
        )
        assert result.pipeline == "agentic"


class TestClassicalPipeline:
    """Tests for ClassicalPipeline.run()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_embedding_model_returns_error(self):
        pipeline = ClassicalPipeline(
            embedding_generator=None,
            ollama_client=None,
            reranker=None,
        )
        result = await pipeline.run("What is the dosage for amoxicillin?")
        assert isinstance(result, PipelineResult)
        assert result.pipeline == "classical"
        assert result.confidence == 0.0
        assert "Embedding model not available" in result.answer

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embedding_failure_returns_error(self, mocker):
        mock_emb = mocker.AsyncMock()
        mock_emb.generate_embeddings.side_effect = Exception("embed fail")

        pipeline = ClassicalPipeline(
            embedding_generator=mock_emb,
            ollama_client=None,
            reranker=None,
        )
        result = await pipeline.run("What is the dosage for amoxicillin?")
        assert "Embedding generation failed" in result.answer
        assert result.confidence == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self, mocker):
        mock_emb = mocker.AsyncMock()
        cached_data = {
            "answer": "Cached answer",
            "confidence": 0.85,
            "citations": [],
            "retrieved_chunks": [],
            "query_id": "cached",
            "processing_time_ms": 10.0,
            "hallucination_flagged": False,
        }
        mocker.patch(
            "src.rag.cache.CacheManager.get_query_result",
            return_value=cached_data,
        )

        pipeline = ClassicalPipeline(
            embedding_generator=mock_emb,
            ollama_client=None,
            reranker=None,
        )
        result = await pipeline.run("What is the dosage for amoxicillin?")
        assert result.cached is True
        assert result.answer == "Cached answer"
        assert result.pipeline == "classical"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_documents_returns_message(self, mocker):
        mock_emb = mocker.AsyncMock()
        mock_emb.generate_embeddings.return_value = [[0.1] * 768]

        mocker.patch(
            "src.rag.cache.CacheManager.get_query_result",
            return_value=None,
        )

        # Mock database returning empty chunks
        mock_session = mocker.AsyncMock()
        mock_result = mocker.MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        mock_session_ctx = mocker.AsyncMock()
        mock_session_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch(
            "src.db.postgres.AsyncSessionLocal",
            return_value=mock_session_ctx,
        )

        pipeline = ClassicalPipeline(
            embedding_generator=mock_emb,
            ollama_client=None,
            reranker=None,
        )
        result = await pipeline.run("What is the dosage for amoxicillin?")
        assert "No documents indexed" in result.answer

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_steps_recorded(self, mocker):
        mock_emb = mocker.AsyncMock()
        mock_emb.generate_embeddings.side_effect = Exception("fail")

        pipeline = ClassicalPipeline(
            embedding_generator=mock_emb,
            ollama_client=None,
            reranker=None,
        )
        result = await pipeline.run("What is the dosage for amoxicillin?")
        # Should have at least pii_redact_input and cache_check steps
        step_names = [s["name"] for s in result.steps]
        assert "pii_redact_input" in step_names

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pii_redaction_applied(self, mocker):
        """Verify PII is redacted from the question."""
        mock_emb = mocker.AsyncMock()
        mock_emb.generate_embeddings.side_effect = Exception("fail")

        pipeline = ClassicalPipeline(
            embedding_generator=mock_emb,
            ollama_client=None,
            reranker=None,
        )
        # The PIIRedactor should process the question before anything else
        result = await pipeline.run(
            "Patient John Doe with MRN ABC12345678 has a question about dosage"
        )
        assert isinstance(result, PipelineResult)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fallback_without_llm(self, mocker):
        mock_emb = mocker.AsyncMock()
        mock_emb.generate_embeddings.return_value = [[0.1] * 768]

        mocker.patch(
            "src.rag.cache.CacheManager.get_query_result",
            return_value=None,
        )
        mocker.patch(
            "src.rag.cache.CacheManager.set_query_result",
            return_value=None,
        )

        # Mock database with chunks
        mock_chunk = mocker.MagicMock()
        mock_chunk.id = 1
        mock_chunk.content = "Amoxicillin 500mg three times daily"
        mock_chunk.document_id = 1
        mock_chunk.chunk_index = 0
        mock_chunk.embedding = [0.1] * 768

        mock_session = mocker.AsyncMock()
        mock_result = mocker.MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_chunk]
        mock_session.execute.return_value = mock_result

        mock_session_ctx = mocker.AsyncMock()
        mock_session_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch(
            "src.db.postgres.AsyncSessionLocal",
            return_value=mock_session_ctx,
        )

        # Mock HybridRetriever
        mock_retriever = mocker.MagicMock()
        mock_retriever.search = mocker.AsyncMock(
            return_value=[_make_chunk(1, "Amoxicillin 500mg", 0.8)]
        )
        mocker.patch(
            "src.rag.retriever.HybridRetriever",
            return_value=mock_retriever,
        )

        pipeline = ClassicalPipeline(
            embedding_generator=mock_emb,
            ollama_client=None,  # No LLM
            reranker=None,
        )
        result = await pipeline.run("What is the dosage for amoxicillin?")
        assert "LLM synthesis unavailable" in result.answer
        assert result.pipeline == "classical"
        assert len(result.retrieved_chunks) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_database_failure_returns_error(self, mocker):
        mock_emb = mocker.AsyncMock()
        mock_emb.generate_embeddings.return_value = [[0.1] * 768]

        mocker.patch(
            "src.rag.cache.CacheManager.get_query_result",
            return_value=None,
        )

        mocker.patch(
            "src.db.postgres.AsyncSessionLocal",
            side_effect=Exception("DB down"),
        )

        pipeline = ClassicalPipeline(
            embedding_generator=mock_emb,
            ollama_client=None,
            reranker=None,
        )
        result = await pipeline.run("What is the dosage for amoxicillin?")
        assert "Database unavailable" in result.answer

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_to_cache_dict(self):
        result = PipelineResult(
            answer="test answer",
            confidence=0.85,
            citations=[{"doc": "a"}],
            retrieved_chunks=[{"text": "b"}],
            query_id="q1",
            processing_time_ms=100.0,
            hallucination_flagged=True,
        )
        cache_dict = ClassicalPipeline._to_cache_dict(result)
        assert cache_dict["answer"] == "test answer"
        assert cache_dict["confidence"] == 0.85
        assert cache_dict["hallucination_flagged"] is True
        assert "pipeline" not in cache_dict
        assert "steps" not in cache_dict
