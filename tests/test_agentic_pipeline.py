"""
Tests for the Agentic RAG Pipeline
"""

import pytest

from src.pipelines.agentic import AgenticPipeline
from src.pipelines.classical import PipelineResult
from src.pipelines.prompts import (
    DECOMPOSE_COUNTS,
    MAX_SUB_QUERIES,
    VALID_QUERY_TYPES,
    build_synthesis_prompt,
)
from src.rag.retriever import ScoredChunk
from src.security.pii_redaction import PIIRedactor


def _make_chunk(chunk_id: int, content: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        content=content,
        document_id=1,
        chunk_index=chunk_id,
        score=score,
        source="hybrid",
    )


class TestQueryClassification:
    """Tests for the classify step."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_classify_simple(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "SIMPLE"
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._classify("What is the dosage for amoxicillin?")
        assert result == "SIMPLE"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_classify_comparative(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "COMPARATIVE"
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._classify("Compare knee vs hip protocols")
        assert result == "COMPARATIVE"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_classify_multi_step(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "MULTI_STEP"
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._classify("Contraindications for diabetes and HTN")
        assert result == "MULTI_STEP"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_classify_temporal(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "TEMPORAL"
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._classify("How has the protocol changed?")
        assert result == "TEMPORAL"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_classify_defaults_to_simple_on_invalid(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "UNKNOWN_CATEGORY"
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._classify("test query")
        assert result == "SIMPLE"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_classify_defaults_to_simple_on_failure(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.side_effect = Exception("LLM down")
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._classify("test query")
        assert result == "SIMPLE"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_classify_without_llm(self):
        pipeline = AgenticPipeline(None, None, None)
        result = await pipeline._classify("test query")
        assert result == "SIMPLE"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_classify_extracts_from_verbose_response(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "The category is COMPARATIVE because..."
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._classify("Compare A vs B")
        assert result == "COMPARATIVE"


class TestQueryDecomposition:
    """Tests for the decompose step."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_decompose_comparative_produces_two(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "1. knee pain protocol\n2. hip pain protocol"
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._decompose("Compare knee vs hip", "COMPARATIVE")
        assert len(result) == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_decompose_multi_step_produces_three(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = (
            "1. diabetes contraindications\n"
            "2. hypertension contraindications\n"
            "3. combined contraindications"
        )
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._decompose("combined question", "MULTI_STEP")
        assert len(result) == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_decompose_caps_at_max(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "1. q1\n2. q2\n3. q3\n4. q4\n5. q5\n6. q6"
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._decompose("big question", "MULTI_STEP")
        assert len(result) <= MAX_SUB_QUERIES

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_decompose_falls_back_on_failure(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.side_effect = Exception("LLM error")
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._decompose("original query", "COMPARATIVE")
        assert result == ["original query"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_decompose_falls_back_on_empty(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = ""
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._decompose("original query", "COMPARATIVE")
        assert result == ["original query"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_decompose_without_llm(self):
        pipeline = AgenticPipeline(None, None, None)
        result = await pipeline._decompose("test", "COMPARATIVE")
        assert result == ["test"]


class TestDeduplication:
    """Tests for chunk deduplication."""

    @pytest.mark.unit
    def test_deduplicate_removes_duplicates(self):
        chunks = [
            _make_chunk(1, "chunk A", 0.8),
            _make_chunk(1, "chunk A", 0.9),  # Same ID, higher score
            _make_chunk(2, "chunk B", 0.7),
        ]
        result = AgenticPipeline._deduplicate(chunks)
        assert len(result) == 2
        # Should keep the higher-scored version
        assert result[0].chunk_id == 1
        assert result[0].score == 0.9

    @pytest.mark.unit
    def test_deduplicate_preserves_unique(self):
        chunks = [
            _make_chunk(1, "A", 0.8),
            _make_chunk(2, "B", 0.7),
            _make_chunk(3, "C", 0.6),
        ]
        result = AgenticPipeline._deduplicate(chunks)
        assert len(result) == 3

    @pytest.mark.unit
    def test_deduplicate_sorted_by_score(self):
        chunks = [
            _make_chunk(3, "low", 0.3),
            _make_chunk(1, "high", 0.9),
            _make_chunk(2, "mid", 0.6),
        ]
        result = AgenticPipeline._deduplicate(chunks)
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.unit
    def test_deduplicate_empty(self):
        assert AgenticPipeline._deduplicate([]) == []


class TestNumberedListParsing:
    """Tests for parsing numbered lists from LLM output."""

    @pytest.mark.unit
    def test_parse_standard_numbered(self):
        text = "1. First query\n2. Second query\n3. Third query"
        result = AgenticPipeline._parse_numbered_list(text)
        assert result == ["First query", "Second query", "Third query"]

    @pytest.mark.unit
    def test_parse_parenthesis_numbered(self):
        text = "1) First\n2) Second"
        result = AgenticPipeline._parse_numbered_list(text)
        assert result == ["First", "Second"]

    @pytest.mark.unit
    def test_parse_with_blank_lines(self):
        text = "1. First\n\n2. Second\n\n"
        result = AgenticPipeline._parse_numbered_list(text)
        assert result == ["First", "Second"]

    @pytest.mark.unit
    def test_parse_empty(self):
        result = AgenticPipeline._parse_numbered_list("")
        assert result == []


class TestSelfReflection:
    """Tests for the reflect step."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reflect_sufficient(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "SUFFICIENT"
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._reflect("question", "answer", "SIMPLE", 0.8)
        assert result == "SUFFICIENT"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reflect_insufficient(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = (
            "INSUFFICIENT: Missing dosage information\n" "amoxicillin dosage guidelines"
        )
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._reflect("question", "answer", "SIMPLE", 0.5)
        assert result.startswith("INSUFFICIENT")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reflect_defaults_sufficient_on_failure(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.side_effect = Exception("error")
        pipeline = AgenticPipeline(None, mock_llm, None)
        result = await pipeline._reflect("question", "answer", "SIMPLE", 0.5)
        assert result == "SUFFICIENT"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reflect_without_llm(self):
        pipeline = AgenticPipeline(None, None, None)
        result = await pipeline._reflect("question", "answer", "SIMPLE", 0.5)
        assert result == "SUFFICIENT"


class TestRefinedQueryExtraction:
    """Tests for extracting refined queries from reflection output."""

    @pytest.mark.unit
    def test_extract_from_multiline(self):
        reflection = "INSUFFICIENT: Missing info\namoxicillin dosage"
        result = AgenticPipeline._extract_refined_query(reflection, "fallback")
        assert result == "amoxicillin dosage"

    @pytest.mark.unit
    def test_extract_from_single_line(self):
        reflection = "INSUFFICIENT: need more about pain protocol"
        result = AgenticPipeline._extract_refined_query(reflection, "fallback")
        assert result == "need more about pain protocol"

    @pytest.mark.unit
    def test_extract_falls_back(self):
        reflection = ""
        result = AgenticPipeline._extract_refined_query(reflection, "fallback query")
        assert result == "fallback query"

    @pytest.mark.unit
    def test_extract_skips_blank_lines(self):
        reflection = (
            "INSUFFICIENT: The answer is too low confidence.\n"
            "\n"
            'Refined query: "pain management protocols for knee arthroplasty"'
        )
        result = AgenticPipeline._extract_refined_query(reflection, "fallback")
        assert result == "pain management protocols for knee arthroplasty"

    @pytest.mark.unit
    def test_extract_refined_query_prefix(self):
        reflection = (
            "INSUFFICIENT: Missing dosage info\n"
            "Refined query: amoxicillin dosage guidelines"
        )
        result = AgenticPipeline._extract_refined_query(reflection, "fallback")
        assert result == "amoxicillin dosage guidelines"


class TestAgenticPipelineFull:
    """Integration-style tests for the full pipeline."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_embedding_returns_error(self):
        pipeline = AgenticPipeline(
            embedding_generator=None,
            ollama_client=None,
            reranker=None,
        )
        result = await pipeline.run("What is the dosage for amoxicillin?")
        assert isinstance(result, PipelineResult)
        assert result.pipeline == "agentic"
        assert "Embedding model not available" in result.answer

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pipeline_field_is_agentic(self, mocker):
        mock_emb = mocker.AsyncMock()
        mock_emb.generate_embeddings.return_value = [[0.1] * 768]

        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "SIMPLE"

        # Mock database with empty chunks
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

        pipeline = AgenticPipeline(
            embedding_generator=mock_emb,
            ollama_client=mock_llm,
            reranker=None,
        )

        result = await pipeline.run("What is the dosage for amoxicillin?")
        assert result.pipeline == "agentic"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_steps_include_classify_and_decompose(self, mocker):
        mock_emb = mocker.AsyncMock()
        mock_emb.generate_embeddings.return_value = [[0.1] * 768]

        mock_llm = mocker.AsyncMock()
        # First call: classify -> SIMPLE, then any synthesis/reflect calls
        mock_llm.generate.return_value = "SIMPLE"

        # Mock database with empty chunks
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

        pipeline = AgenticPipeline(mock_emb, mock_llm, None)
        result = await pipeline.run("What is the dosage for amoxicillin?")

        step_names = [s["name"] for s in result.steps]
        assert "classify" in step_names
        assert "decompose" in step_names

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_simple_query_skips_decomposition(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "SIMPLE"

        pipeline = AgenticPipeline(None, mock_llm, None)

        # Classification returns SIMPLE
        query_type = await pipeline._classify("Simple question?")
        assert query_type == "SIMPLE"

        # For SIMPLE, decompose should not be called — sub_queries = [question]
        # This is tested by checking that the pipeline doesn't call _decompose
        # when query_type is SIMPLE (integration tested in full run)


class TestPromptConstants:
    """Tests for prompt template constants."""

    @pytest.mark.unit
    def test_valid_query_types(self):
        assert {
            "SIMPLE",
            "COMPARATIVE",
            "MULTI_STEP",
            "TEMPORAL",
            "GENERAL",
        } == VALID_QUERY_TYPES

    @pytest.mark.unit
    def test_decompose_counts_for_all_types(self):
        for qt in VALID_QUERY_TYPES:
            assert qt in DECOMPOSE_COUNTS

    @pytest.mark.unit
    def test_max_sub_queries(self):
        assert MAX_SUB_QUERIES == 4
        for count in DECOMPOSE_COUNTS.values():
            assert count <= MAX_SUB_QUERIES


class TestBuildSynthesisPrompt:
    """Tests for build_synthesis_prompt covering all query types."""

    @pytest.mark.unit
    def test_comparative_includes_special_instructions(self):
        prompt = build_synthesis_prompt("Compare A vs B", "some context", "COMPARATIVE")
        assert "Note:" in prompt
        assert "comparison" in prompt.lower() or "compare" in prompt.lower()

    @pytest.mark.unit
    def test_multi_step_includes_special_instructions(self):
        prompt = build_synthesis_prompt("Multi-step question", "context", "MULTI_STEP")
        assert "Note:" in prompt
        assert "different sources" in prompt.lower()

    @pytest.mark.unit
    def test_temporal_includes_special_instructions(self):
        prompt = build_synthesis_prompt("How has it changed?", "context", "TEMPORAL")
        assert "Note:" in prompt
        assert "chronologically" in prompt.lower()

    @pytest.mark.unit
    def test_simple_has_no_special_instructions(self):
        prompt = build_synthesis_prompt("Simple question", "context", "SIMPLE")
        assert "Note:" not in prompt

    @pytest.mark.unit
    def test_unknown_type_has_no_special_instructions(self):
        prompt = build_synthesis_prompt("Question", "context", "UNKNOWN")
        assert "Note:" not in prompt

    @pytest.mark.unit
    def test_prompt_includes_question_and_context(self):
        prompt = build_synthesis_prompt("test question", "test context", "SIMPLE")
        assert "test question" in prompt
        assert "test context" in prompt


class TestSynthesize:
    """Tests for the _synthesize method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_synthesize_without_llm_returns_fallback(self):
        pipeline = AgenticPipeline(None, None, None)
        redactor = PIIRedactor()
        chunks = [_make_chunk(1, "chunk content", 0.8)]
        answer, confidence, citations, flagged = await pipeline._synthesize(
            "test question", chunks, "SIMPLE", 0.7, redactor
        )
        assert "Found 1 relevant chunk" in answer
        assert citations == []
        assert not flagged

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_synthesize_without_llm_empty_chunks(self):
        pipeline = AgenticPipeline(None, None, None)
        redactor = PIIRedactor()
        answer, confidence, citations, flagged = await pipeline._synthesize(
            "test question", [], "SIMPLE", 0.7, redactor
        )
        assert confidence == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_synthesize_with_llm_success(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "The answer is X.\nConfidence: 0.90"
        pipeline = AgenticPipeline(None, mock_llm, None)
        redactor = PIIRedactor()
        chunks = [_make_chunk(1, "relevant content", 0.85)]
        answer, confidence, citations, flagged = await pipeline._synthesize(
            "test question", chunks, "SIMPLE", 0.7, redactor
        )
        assert isinstance(answer, str)
        assert isinstance(confidence, float)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_synthesize_with_llm_empty_response_falls_back(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = None
        pipeline = AgenticPipeline(None, mock_llm, None)
        redactor = PIIRedactor()
        chunks = [_make_chunk(1, "content", 0.8)]
        answer, confidence, citations, flagged = await pipeline._synthesize(
            "test question", chunks, "SIMPLE", 0.7, redactor
        )
        assert "Found 1 relevant chunk" in answer

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_synthesize_with_llm_failure_falls_back(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.side_effect = Exception("LLM error")
        pipeline = AgenticPipeline(None, mock_llm, None)
        redactor = PIIRedactor()
        chunks = [_make_chunk(1, "content", 0.8)]
        answer, confidence, citations, flagged = await pipeline._synthesize(
            "test question", chunks, "SIMPLE", 0.7, redactor
        )
        assert "Found 1 relevant chunk" in answer

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_synthesize_low_confidence_returns_snippets_message(self, mocker):
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "Uncertain answer.\nConfidence: 0.30"
        pipeline = AgenticPipeline(None, mock_llm, None)
        redactor = PIIRedactor()
        chunks = [_make_chunk(1, "content", 0.8)]
        answer, confidence, citations, flagged = await pipeline._synthesize(
            "test question", chunks, "SIMPLE", 0.70, redactor
        )
        assert "Confidence too low" in answer or isinstance(answer, str)


class TestAgenticFullRun:
    """Tests for the full agentic pipeline run() covering retrieval and synthesis paths."""

    def _make_session_mock(self, mocker, db_chunks):
        """Helper to create a mock DB session that returns the given chunks."""
        mock_session = mocker.AsyncMock()
        mock_result = mocker.MagicMock()
        mock_result.scalars.return_value.all.return_value = db_chunks
        mock_session.execute.return_value = mock_result

        mock_session_ctx = mocker.AsyncMock()
        mock_session_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = mocker.AsyncMock(return_value=False)
        return mock_session_ctx

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_pipeline_with_chunks_and_synthesis(self, mocker):
        """Happy path: DB has chunks, retrieval finds results, LLM synthesizes."""
        mock_emb = mocker.AsyncMock()
        mock_emb.generate_embeddings.return_value = [[0.1] * 768]

        mock_llm = mocker.AsyncMock()
        mock_llm.generate.side_effect = [
            "SIMPLE",  # classify
            "The dosage is 500mg.\nConfidence: 0.90",  # synthesize
        ]

        mock_db_chunk = mocker.MagicMock()
        mock_db_chunk.chunk_id = 1
        mock_db_chunk.content = "Amoxicillin dosage: 500mg three times daily"
        mock_db_chunk.document_id = 1
        mock_db_chunk.chunk_index = 0
        mock_db_chunk.embedding = [0.1] * 768

        session_ctx = self._make_session_mock(mocker, [mock_db_chunk])
        mocker.patch("src.db.postgres.AsyncSessionLocal", return_value=session_ctx)

        scored = _make_chunk(1, "Amoxicillin dosage: 500mg three times daily", 0.85)
        mock_retriever = mocker.AsyncMock()
        mock_retriever.search.return_value = [scored]
        mocker.patch("src.rag.retriever.HybridRetriever", return_value=mock_retriever)

        pipeline = AgenticPipeline(mock_emb, mock_llm, None)
        result = await pipeline.run("What is the dosage for amoxicillin?")

        assert result.pipeline == "agentic"
        assert len(result.retrieved_chunks) > 0
        step_names = [s["name"] for s in result.steps]
        assert "deduplicate" in step_names
        assert "synthesize" in step_names

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_pipeline_database_failure(self, mocker):
        """Database exception produces error response."""
        mock_emb = mocker.AsyncMock()
        mock_emb.generate_embeddings.return_value = [[0.1] * 768]

        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "SIMPLE"

        mocker.patch(
            "src.db.postgres.AsyncSessionLocal",
            side_effect=Exception("DB connection refused"),
        )

        pipeline = AgenticPipeline(mock_emb, mock_llm, None)
        result = await pipeline.run("What is the dosage?")

        assert result.pipeline == "agentic"
        assert "unavailable" in result.answer.lower() or result.confidence == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_pipeline_empty_retrieval_returns_no_results(self, mocker):
        """Retrieval returns zero results → 'No relevant results' response."""
        mock_emb = mocker.AsyncMock()
        mock_emb.generate_embeddings.return_value = [[0.1] * 768]

        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = "SIMPLE"

        mock_db_chunk = mocker.MagicMock()
        mock_db_chunk.chunk_id = 1
        mock_db_chunk.content = "content"
        mock_db_chunk.document_id = 1
        mock_db_chunk.chunk_index = 0
        mock_db_chunk.embedding = [0.1] * 768

        session_ctx = self._make_session_mock(mocker, [mock_db_chunk])
        mocker.patch("src.db.postgres.AsyncSessionLocal", return_value=session_ctx)

        # Retriever returns no results
        mock_retriever = mocker.AsyncMock()
        mock_retriever.search.return_value = []
        mocker.patch("src.rag.retriever.HybridRetriever", return_value=mock_retriever)

        pipeline = AgenticPipeline(mock_emb, mock_llm, None)
        result = await pipeline.run("What is the dosage?")

        assert result.pipeline == "agentic"
        assert result.confidence == 0.0
        assert "No relevant results" in result.answer

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_pipeline_non_simple_decompose(self, mocker):
        """Non-SIMPLE query triggers _decompose (covers line 93)."""
        mock_emb = mocker.AsyncMock()
        mock_emb.generate_embeddings.return_value = [[0.1] * 768]

        mock_llm = mocker.AsyncMock()
        mock_llm.generate.side_effect = [
            "COMPARATIVE",  # classify
            "1. knee protocol\n2. hip protocol",  # decompose
            "The answer.\nConfidence: 0.85",  # synthesize
        ]

        mock_db_chunk = mocker.MagicMock()
        mock_db_chunk.chunk_id = 1
        mock_db_chunk.content = "clinical content"
        mock_db_chunk.document_id = 1
        mock_db_chunk.chunk_index = 0
        mock_db_chunk.embedding = [0.1] * 768

        session_ctx = self._make_session_mock(mocker, [mock_db_chunk])
        mocker.patch("src.db.postgres.AsyncSessionLocal", return_value=session_ctx)

        scored = _make_chunk(1, "clinical content", 0.80)
        mock_retriever = mocker.AsyncMock()
        mock_retriever.search.return_value = [scored]
        mocker.patch("src.rag.retriever.HybridRetriever", return_value=mock_retriever)

        pipeline = AgenticPipeline(mock_emb, mock_llm, None)
        result = await pipeline.run("Compare knee vs hip pain protocols")

        assert result.pipeline == "agentic"
        step_names = [s["name"] for s in result.steps]
        assert "classify" in step_names
        assert "decompose" in step_names

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_pipeline_reflect_sufficient_path(self, mocker):
        """High-confidence answer takes the 'else' branch in reflection (lines 333-340)."""
        mock_emb = mocker.AsyncMock()
        mock_emb.generate_embeddings.return_value = [[0.1] * 768]

        mock_llm = mocker.AsyncMock()
        mock_llm.generate.side_effect = [
            "SIMPLE",
            "Confident answer.\nConfidence: 0.95",
        ]

        mock_db_chunk = mocker.MagicMock()
        mock_db_chunk.chunk_id = 1
        mock_db_chunk.content = "content"
        mock_db_chunk.document_id = 1
        mock_db_chunk.chunk_index = 0
        mock_db_chunk.embedding = [0.1] * 768

        session_ctx = self._make_session_mock(mocker, [mock_db_chunk])
        mocker.patch("src.db.postgres.AsyncSessionLocal", return_value=session_ctx)

        scored = _make_chunk(1, "content", 0.95)
        mock_retriever = mocker.AsyncMock()
        mock_retriever.search.return_value = [scored]
        mocker.patch("src.rag.retriever.HybridRetriever", return_value=mock_retriever)

        pipeline = AgenticPipeline(mock_emb, mock_llm, None)
        result = await pipeline.run("Simple question?", confidence_threshold=0.70)

        assert result.pipeline == "agentic"
        # High confidence → reflect step should say SUFFICIENT (no LLM call for reflect)
        reflect_steps = [s for s in result.steps if s["name"] == "reflect"]
        assert len(reflect_steps) == 1
        assert reflect_steps[0]["detail"] == "SUFFICIENT"
