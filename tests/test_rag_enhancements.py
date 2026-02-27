"""
Tests for RAG Enhancement Functions

Covers:
- Multi-query retrieval (generate_query_variants, extract_medical_entities, boost_entity_matches)
- Context window expansion (expand_context_window)
- Sentence-level extraction (extract_key_sentences, build_extractive_answer, _split_sentences)
- Web search fallback (search_web, format_web_results_as_context, web_results_to_chunks)
- Conditional routing (_direct_answer)
"""

import pytest

from src.pipelines.agentic import AgenticPipeline
from src.rag.reranker import (
    RerankedChunk,
    _split_sentences,
    build_extractive_answer,
    extract_key_sentences,
)
from src.rag.retriever import (
    ScoredChunk,
    boost_entity_matches,
    expand_context_window,
    extract_medical_entities,
    generate_query_variants,
)
from src.rag.web_search import (
    WebResult,
    format_web_results_as_context,
    search_web,
    web_results_to_chunks,
)

# ============================================
# Helpers
# ============================================


def _make_scored_chunk(
    chunk_id: int,
    content: str,
    score: float,
    document_id: int = 1,
    chunk_index: int = 0,
    source: str = "hybrid",
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        content=content,
        document_id=document_id,
        chunk_index=chunk_index,
        score=score,
        source=source,
    )


def _make_reranked_chunk(
    chunk_id: int,
    content: str,
    final_score: float,
    document_id: int = 1,
    chunk_index: int = 0,
    source: str = "hybrid",
) -> RerankedChunk:
    return RerankedChunk(
        chunk_id=chunk_id,
        content=content,
        document_id=document_id,
        chunk_index=chunk_index,
        retrieval_score=final_score,
        rerank_score=final_score,
        final_score=final_score,
        source=source,
    )


# ============================================
# Multi-Query Retrieval: generate_query_variants
# ============================================


class TestGenerateQueryVariants:
    """Tests for generate_query_variants()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_with_mock_ollama_returns_variants(self, mocker):
        """LLM generates numbered variants; result = [original] + variants."""
        mock_client = mocker.AsyncMock()
        mock_client.generate.return_value = (
            "1. What is the recommended dose of amoxicillin?\n"
            "2. Amoxicillin dosing guidelines\n"
            "3. How much amoxicillin should be prescribed?"
        )
        query = "What is the dosage for amoxicillin?"
        result = await generate_query_variants(query, mock_client, n=3)

        assert result[0] == query
        assert len(result) >= 2  # original + at least 1 variant
        assert len(result) <= 4  # original + up to 3 variants
        mock_client.generate.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_without_client_returns_original_only(self):
        """No ollama_client => fallback to [original query]."""
        result = await generate_query_variants("test query", None, n=3)
        assert result == ["test query"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_returns_empty_string(self, mocker):
        """LLM returns empty string => fallback to [original query]."""
        mock_client = mocker.AsyncMock()
        mock_client.generate.return_value = ""
        result = await generate_query_variants("test query", mock_client, n=3)
        assert result == ["test query"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_returns_none(self, mocker):
        """LLM returns None => fallback to [original query]."""
        mock_client = mocker.AsyncMock()
        mock_client.generate.return_value = None
        result = await generate_query_variants("test query", mock_client, n=3)
        assert result == ["test query"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_exception_returns_original(self, mocker):
        """LLM raises exception => fallback to [original query]."""
        mock_client = mocker.AsyncMock()
        mock_client.generate.side_effect = Exception("LLM timeout")
        result = await generate_query_variants("test query", mock_client, n=3)
        assert result == ["test query"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_deduplicates_if_variant_matches_original(self, mocker):
        """Variants matching the original query (case-insensitive) are excluded."""
        mock_client = mocker.AsyncMock()
        mock_client.generate.return_value = (
            "1. What is the dosage for amoxicillin?\n"  # matches original
            "2. Amoxicillin dosing info"
        )
        query = "What is the dosage for amoxicillin?"
        result = await generate_query_variants(query, mock_client, n=2)
        assert result[0] == query
        # The duplicate of the original should be filtered out
        lowered = [r.lower() for r in result]
        assert lowered.count(query.lower()) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_respects_n_limit(self, mocker):
        """At most n variants are returned (plus the original)."""
        mock_client = mocker.AsyncMock()
        mock_client.generate.return_value = (
            "1. Variant A\n2. Variant B\n3. Variant C\n4. Variant D\n5. Variant E"
        )
        result = await generate_query_variants("original", mock_client, n=2)
        # original + at most 2 variants
        assert len(result) <= 3


# ============================================
# Multi-Query Retrieval: extract_medical_entities
# ============================================


class TestExtractMedicalEntities:
    """Tests for extract_medical_entities()."""

    @pytest.mark.unit
    def test_detects_abbreviation(self):
        """Known abbreviations (e.g., MI) are extracted along with expansions."""
        entities = extract_medical_entities("Patient has MI and CHF")
        lower_entities = [e.lower() for e in entities]
        assert "mi" in lower_entities
        assert "myocardial infarction" in lower_entities
        assert "chf" in lower_entities
        assert "congestive heart failure" in lower_entities

    @pytest.mark.unit
    def test_detects_drug_suffix(self):
        """Drug-like terms ending in known suffixes are detected."""
        entities = extract_medical_entities("Prescribe Amoxicillin and Atorvastatin")
        lower_entities = [e.lower() for e in entities]
        # Amoxicillin matches -cillin suffix; Atorvastatin matches -statin suffix
        assert any("cillin" in e for e in lower_entities)
        assert any("statin" in e for e in lower_entities)

    @pytest.mark.unit
    def test_detects_dosage_pattern(self):
        """Dosage patterns like '500 mg' are extracted."""
        entities = extract_medical_entities("Give 500 mg ibuprofen every 6 hours")
        lower_entities = [e.lower() for e in entities]
        assert any("500" in e and "mg" in e for e in lower_entities)

    @pytest.mark.unit
    def test_detects_procedure(self):
        """Common procedure terms are detected."""
        entities = extract_medical_entities("Schedule a colonoscopy for next week")
        lower_entities = [e.lower() for e in entities]
        assert "colonoscopy" in lower_entities

    @pytest.mark.unit
    def test_empty_query_returns_empty(self):
        """Empty string returns empty list."""
        entities = extract_medical_entities("")
        assert entities == []

    @pytest.mark.unit
    def test_no_entities_returns_empty(self):
        """Query with no medical entities returns empty list."""
        entities = extract_medical_entities("hello world how are you today")
        assert entities == []

    @pytest.mark.unit
    def test_returns_unique_entities(self):
        """Duplicate entities are deduplicated."""
        entities = extract_medical_entities("MI MI MI")
        # Should still contain 'mi' and 'myocardial infarction' but not duplicates
        assert len(entities) == len(set(entities))


# ============================================
# Multi-Query Retrieval: boost_entity_matches
# ============================================


class TestBoostEntityMatches:
    """Tests for boost_entity_matches()."""

    @pytest.mark.unit
    def test_boosts_matching_chunks(self):
        """Chunks containing entities get higher scores."""
        chunks = [
            _make_scored_chunk(1, "Patient has myocardial infarction history", 0.5),
            _make_scored_chunk(2, "Routine checkup notes", 0.5),
        ]
        entities = ["myocardial infarction"]
        result = boost_entity_matches(chunks, entities, boost_factor=1.15)

        # The chunk with the entity should have a higher score
        scores_by_id = {c.chunk_id: c.score for c in result}
        assert scores_by_id[1] > scores_by_id[2]

    @pytest.mark.unit
    def test_non_matching_chunks_unchanged(self):
        """Chunks without entity matches keep their original score."""
        chunks = [
            _make_scored_chunk(1, "Routine checkup notes", 0.5),
        ]
        entities = ["myocardial infarction"]
        result = boost_entity_matches(chunks, entities)

        assert result[0].score == 0.5
        assert result[0].chunk_id == 1

    @pytest.mark.unit
    def test_empty_entities_returns_original(self):
        """No entities => chunks returned unchanged."""
        chunks = [_make_scored_chunk(1, "content", 0.5)]
        result = boost_entity_matches(chunks, [])
        assert result == chunks

    @pytest.mark.unit
    def test_empty_chunks_returns_empty(self):
        """No chunks => empty list returned."""
        result = boost_entity_matches([], ["entity"])
        assert result == []

    @pytest.mark.unit
    def test_multiple_entity_matches_get_higher_boost(self):
        """Chunks matching more entities get progressively higher boosts."""
        chunks = [
            _make_scored_chunk(1, "Patient has MI and CHF and DVT", 0.5),
            _make_scored_chunk(2, "Patient has MI only", 0.5),
        ]
        entities = ["mi", "chf", "dvt"]
        result = boost_entity_matches(chunks, entities, boost_factor=1.15)

        scores_by_id = {c.chunk_id: c.score for c in result}
        # Chunk 1 matches 3 entities, chunk 2 matches 1 => chunk 1 should score higher
        assert scores_by_id[1] > scores_by_id[2]

    @pytest.mark.unit
    def test_score_capped_at_1(self):
        """Boosted scores should never exceed 1.0."""
        chunks = [
            _make_scored_chunk(1, "mi chf dvt", 0.95),
        ]
        entities = ["mi", "chf", "dvt"]
        result = boost_entity_matches(chunks, entities, boost_factor=1.5)
        assert result[0].score <= 1.0

    @pytest.mark.unit
    def test_result_sorted_by_score_descending(self):
        """Results are sorted by score descending."""
        chunks = [
            _make_scored_chunk(1, "no match", 0.3),
            _make_scored_chunk(2, "myocardial infarction", 0.5),
            _make_scored_chunk(3, "myocardial infarction and hypertension", 0.4),
        ]
        entities = ["myocardial infarction", "hypertension"]
        result = boost_entity_matches(chunks, entities)
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)


# ============================================
# Context Window Expansion: expand_context_window
# ============================================


class TestExpandContextWindow:
    """Tests for expand_context_window()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_expands_with_adjacent_rows(self, mocker):
        """Fetches adjacent chunks from DB and appends them."""
        chunks = [
            _make_scored_chunk(10, "middle chunk", 0.8, document_id=1, chunk_index=2),
        ]

        # Mock the DB session to return adjacent chunk rows
        mock_row = mocker.MagicMock()
        mock_row.id = 11
        mock_row.chunk_text = "adjacent chunk content"
        mock_row.document_id = 1
        mock_row.chunk_index = 3

        mock_result = mocker.MagicMock()
        mock_result.fetchall.return_value = [mock_row]

        mock_session = mocker.AsyncMock()
        mock_session.execute.return_value = mock_result

        result = await expand_context_window(chunks, mock_session, window=1)

        assert len(result) == 2  # original + 1 adjacent
        assert result[0].chunk_id == 10  # original chunk
        assert result[1].chunk_id == 11  # adjacent chunk
        assert result[1].source == "context_expansion"
        # Adjacent chunk gets half of parent's score
        assert result[1].score == pytest.approx(0.4)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty(self, mocker):
        """No chunks to expand => return as-is."""
        mock_session = mocker.AsyncMock()
        result = await expand_context_window([], mock_session, window=1)
        assert result == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_session_error_returns_original_chunks(self, mocker):
        """DB error during expansion => return original chunks unmodified."""
        chunks = [
            _make_scored_chunk(10, "content", 0.8, document_id=1, chunk_index=2),
        ]

        mock_session = mocker.AsyncMock()
        mock_session.execute.side_effect = Exception("DB connection lost")

        result = await expand_context_window(chunks, mock_session, window=1)
        assert len(result) == 1
        assert result[0].chunk_id == 10

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_adjacent_rows_returns_original(self, mocker):
        """DB returns no adjacent rows => return original chunks only."""
        chunks = [
            _make_scored_chunk(10, "content", 0.8, document_id=1, chunk_index=0),
        ]

        mock_result = mocker.MagicMock()
        mock_result.fetchall.return_value = []

        mock_session = mocker.AsyncMock()
        mock_session.execute.return_value = mock_result

        result = await expand_context_window(chunks, mock_session, window=1)
        assert len(result) == 1
        assert result[0].chunk_id == 10

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_does_not_duplicate_existing_ids(self, mocker):
        """Adjacent rows already in the chunk set are not duplicated."""
        chunks = [
            _make_scored_chunk(10, "chunk A", 0.8, document_id=1, chunk_index=1),
            _make_scored_chunk(11, "chunk B", 0.7, document_id=1, chunk_index=2),
        ]

        # DB returns chunk 11 as adjacent to chunk 10, but it's already present
        mock_row = mocker.MagicMock()
        mock_row.id = 11
        mock_row.chunk_text = "chunk B"
        mock_row.document_id = 1
        mock_row.chunk_index = 2

        mock_result = mocker.MagicMock()
        mock_result.fetchall.return_value = [mock_row]

        mock_session = mocker.AsyncMock()
        mock_session.execute.return_value = mock_result

        result = await expand_context_window(chunks, mock_session, window=1)
        chunk_ids = [c.chunk_id for c in result]
        # chunk 11 should appear only once
        assert chunk_ids.count(11) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chunk_at_index_zero_skips_negative_index(self, mocker):
        """Chunk at index 0 should not try to fetch index -1."""
        chunks = [
            _make_scored_chunk(10, "first chunk", 0.8, document_id=1, chunk_index=0),
        ]

        mock_result = mocker.MagicMock()
        mock_result.fetchall.return_value = []

        mock_session = mocker.AsyncMock()
        mock_session.execute.return_value = mock_result

        result = await expand_context_window(chunks, mock_session, window=1)
        # Should still work without errors
        assert len(result) == 1


# ============================================
# Sentence Extraction: _split_sentences
# ============================================


class TestSplitSentences:
    """Tests for _split_sentences()."""

    @pytest.mark.unit
    def test_splits_on_period_followed_by_capital(self):
        """Standard sentence boundary: period + space + capital letter."""
        text = "The patient was admitted. Blood pressure was elevated. Treatment was started."
        sentences = _split_sentences(text)
        assert len(sentences) == 3

    @pytest.mark.unit
    def test_filters_tiny_fragments(self):
        """Fragments <= 20 chars are filtered out."""
        text = "Ok. This is a much longer sentence with enough characters."
        sentences = _split_sentences(text)
        # "Ok." is only 3 chars, should be filtered
        assert all(len(s) > 20 for s in sentences)

    @pytest.mark.unit
    def test_splits_on_newlines(self):
        """Splits on newline boundaries where next line starts with non-whitespace."""
        text = (
            "ASSESSMENT: The patient shows improvement.\n"
            "PLAN: Continue current medications and monitor closely over the next week."
        )
        sentences = _split_sentences(text)
        assert len(sentences) >= 2

    @pytest.mark.unit
    def test_empty_string_returns_empty(self):
        """Empty input returns empty list."""
        assert _split_sentences("") == []

    @pytest.mark.unit
    def test_single_long_sentence(self):
        """A single sentence longer than 20 chars is returned as-is."""
        text = "Administer acetaminophen 1000mg every 6 hours for post-operative pain management."
        sentences = _split_sentences(text)
        assert len(sentences) == 1
        assert "acetaminophen" in sentences[0]


# ============================================
# Sentence Extraction: extract_key_sentences
# ============================================


class TestExtractKeySentences:
    """Tests for extract_key_sentences()."""

    @pytest.mark.unit
    def test_extracts_relevant_sentences(self):
        """Extracts sentences most relevant to the query via BM25."""
        chunks = [
            _make_reranked_chunk(
                1,
                "Administer acetaminophen 1000mg every 6 hours. "
                "Monitor patient vitals regularly. "
                "Aspirin should be avoided in this case.",
                0.9,
            ),
        ]
        sentences = extract_key_sentences(
            chunks, "acetaminophen dosage", max_sentences=2
        )
        assert len(sentences) > 0
        assert len(sentences) <= 2

    @pytest.mark.unit
    def test_empty_chunks_returns_empty(self):
        """No chunks => empty list."""
        sentences = extract_key_sentences([], "some query")
        assert sentences == []

    @pytest.mark.unit
    def test_chunks_with_no_splittable_sentences_returns_content(self):
        """If no sentences pass the length filter, returns chunk content directly."""
        chunks = [
            _make_reranked_chunk(1, "Short.", 0.9),
            _make_reranked_chunk(2, "Tiny.", 0.8),
            _make_reranked_chunk(3, "Also small and brief.", 0.7),
        ]
        result = extract_key_sentences(chunks, "query", max_sentences=5)
        # Should fall back to raw content of up to 3 chunks
        assert len(result) > 0

    @pytest.mark.unit
    def test_respects_max_sentences(self):
        """Returns at most max_sentences results."""
        content = ". ".join(
            [
                f"This is test sentence number {i} with enough characters to pass the filter"
                for i in range(20)
            ]
        )
        chunks = [_make_reranked_chunk(1, content, 0.9)]
        sentences = extract_key_sentences(chunks, "test sentence", max_sentences=3)
        assert len(sentences) <= 3

    @pytest.mark.unit
    def test_multiple_chunks_combined(self):
        """Sentences from multiple chunks are combined and ranked."""
        chunks = [
            _make_reranked_chunk(
                1,
                "Acetaminophen is recommended for pain relief. Ibuprofen is an alternative option.",
                0.9,
            ),
            _make_reranked_chunk(
                2,
                "The patient experienced significant pain post-surgery. Acetaminophen was administered immediately.",
                0.8,
            ),
        ]
        sentences = extract_key_sentences(
            chunks, "acetaminophen pain relief", max_sentences=5
        )
        assert len(sentences) > 0


# ============================================
# Sentence Extraction: build_extractive_answer
# ============================================


class TestBuildExtractiveAnswer:
    """Tests for build_extractive_answer()."""

    @pytest.mark.unit
    def test_builds_formatted_answer(self):
        """Returns a numbered list of sentences with a header."""
        chunks = [
            _make_reranked_chunk(
                1,
                "Administer acetaminophen 1000mg every 6 hours for pain management. "
                "Monitor for adverse reactions at regular intervals.",
                0.9,
            ),
        ]
        answer = build_extractive_answer(
            chunks, "acetaminophen dosage", max_sentences=3
        )
        assert "Based on the most relevant passages" in answer
        assert "1." in answer

    @pytest.mark.unit
    def test_empty_chunks_returns_no_info_message(self):
        """No chunks => 'No relevant information found' message."""
        answer = build_extractive_answer([], "some query")
        assert "No relevant information found" in answer

    @pytest.mark.unit
    def test_respects_max_sentences(self):
        """Output contains at most max_sentences numbered items."""
        content = ". ".join(
            [
                f"This is a long sentence number {i} that should pass the length filter easily"
                for i in range(10)
            ]
        )
        chunks = [_make_reranked_chunk(1, content, 0.9)]
        answer = build_extractive_answer(chunks, "sentence", max_sentences=2)
        # Count numbered items (1. and 2.)
        assert "1." in answer
        assert "2." in answer
        assert "3." not in answer


# ============================================
# Web Search: search_web
# ============================================


class TestSearchWeb:
    """Tests for search_web()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_with_mocked_ddgs(self, mocker):
        """Mocked DDGS returns WebResult objects."""
        mock_ddgs_instance = mocker.MagicMock()
        mock_ddgs_instance.__enter__ = mocker.MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = mocker.MagicMock(return_value=False)
        mock_ddgs_instance.text.return_value = [
            {
                "title": "Pain Management Overview",
                "body": "Comprehensive guide to post-operative pain management.",
                "href": "https://example.com/pain-mgmt",
            },
            {
                "title": "Clinical Guidelines",
                "body": "Evidence-based clinical guidelines for pain relief.",
                "href": "https://example.com/guidelines",
            },
        ]

        mock_ddgs_class = mocker.MagicMock(return_value=mock_ddgs_instance)
        mocker.patch("src.rag.web_search.DDGS", mock_ddgs_class, create=True)
        # Patch the import inside the function
        mocker.patch.dict(
            "sys.modules",
            {"duckduckgo_search": mocker.MagicMock(DDGS=mock_ddgs_class)},
        )

        results = await search_web("pain management protocols", max_results=2)

        assert len(results) == 2
        assert isinstance(results[0], WebResult)
        assert results[0].title == "Pain Management Overview"
        assert results[0].source == "web"
        assert "example.com" in results[1].url

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_import_error_returns_empty(self, mocker):
        """When DDGS import fails inside the function, return empty list."""
        # Simulate ImportError by making the duckduckgo_search module unavailable
        import sys

        saved = sys.modules.pop("duckduckgo_search", None)
        mocker.patch.dict("sys.modules", {"duckduckgo_search": None})
        try:
            # Import the actual function (not mocked) — it does lazy import inside
            import importlib

            from src.rag import web_search

            importlib.reload(web_search)
            results = await web_search.search_web("test query")
            assert results == []
        finally:
            if saved is not None:
                sys.modules["duckduckgo_search"] = saved
            import importlib

            from src.rag import web_search

            importlib.reload(web_search)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ddgs_exception_returns_empty(self, mocker):
        """Runtime exception from DDGS => empty list."""
        mock_ddgs_instance = mocker.MagicMock()
        mock_ddgs_instance.__enter__ = mocker.MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = mocker.MagicMock(return_value=False)
        mock_ddgs_instance.text.side_effect = RuntimeError("Rate limited")

        mock_ddgs_class = mocker.MagicMock(return_value=mock_ddgs_instance)
        mocker.patch.dict(
            "sys.modules",
            {"duckduckgo_search": mocker.MagicMock(DDGS=mock_ddgs_class)},
        )

        results = await search_web("test query")
        assert results == []


# ============================================
# Web Search: format_web_results_as_context
# ============================================


class TestFormatWebResultsAsContext:
    """Tests for format_web_results_as_context()."""

    @pytest.mark.unit
    def test_formats_results_correctly(self):
        """Formats WebResults into labeled context sections."""
        results = [
            WebResult(
                title="Pain Management",
                snippet="A guide to pain management.",
                url="https://example.com/pain",
            ),
            WebResult(
                title="Clinical Protocols",
                snippet="Standard protocols for care.",
                url="https://example.com/protocols",
            ),
        ]
        context = format_web_results_as_context(results)

        assert "[Web Source 1: Pain Management]" in context
        assert "https://example.com/pain" in context
        assert "[Web Source 2: Clinical Protocols]" in context
        assert "A guide to pain management." in context
        assert "Standard protocols for care." in context

    @pytest.mark.unit
    def test_empty_results_returns_empty_string(self):
        """No results => empty string."""
        assert format_web_results_as_context([]) == ""

    @pytest.mark.unit
    def test_single_result(self):
        """Single result is formatted correctly."""
        results = [
            WebResult(
                title="Test",
                snippet="Test snippet text.",
                url="https://test.com",
            ),
        ]
        context = format_web_results_as_context(results)
        assert "[Web Source 1: Test]" in context
        assert "URL: https://test.com" in context
        assert "Test snippet text." in context


# ============================================
# Web Search: web_results_to_chunks
# ============================================


class TestWebResultsToChunks:
    """Tests for web_results_to_chunks()."""

    @pytest.mark.unit
    def test_converts_to_chunk_dicts(self):
        """WebResults become chunk-compatible dicts."""
        results = [
            WebResult(
                title="Title A",
                snippet="Snippet A",
                url="https://a.com",
            ),
            WebResult(
                title="Title B",
                snippet="Snippet B",
                url="https://b.com",
            ),
        ]
        chunks = web_results_to_chunks(results)

        assert len(chunks) == 2
        assert chunks[0]["chunk_id"] == "web_0"
        assert chunks[0]["text"] == "Snippet A"
        assert chunks[0]["document_id"] == "web_https://a.com"
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["relevance_score"] == 0.5
        assert chunks[0]["source"] == "web: Title A"
        assert chunks[0]["url"] == "https://a.com"

        assert chunks[1]["chunk_id"] == "web_1"
        assert chunks[1]["source"] == "web: Title B"

    @pytest.mark.unit
    def test_empty_results_returns_empty(self):
        """No results => empty list."""
        assert web_results_to_chunks([]) == []

    @pytest.mark.unit
    def test_chunk_fields_present(self):
        """All expected fields are present in chunk dicts."""
        results = [
            WebResult(title="T", snippet="S", url="https://u.com"),
        ]
        chunk = web_results_to_chunks(results)[0]
        expected_keys = {
            "chunk_id",
            "text",
            "document_id",
            "chunk_index",
            "relevance_score",
            "source",
            "url",
        }
        assert set(chunk.keys()) == expected_keys


# ============================================
# Conditional Routing: _direct_answer
# ============================================


class TestDirectAnswer:
    """Tests for AgenticPipeline._direct_answer()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_with_mock_llm(self, mocker):
        """LLM generates a direct answer with confidence."""
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = (
            "Hypertension is a condition of elevated blood pressure.\n"
            "Confidence: 0.85"
        )
        pipeline = AgenticPipeline(None, mock_llm, None)
        answer, confidence = await pipeline._direct_answer("What is hypertension?")

        assert "general medical knowledge" in answer.lower()
        assert isinstance(confidence, float)
        assert confidence > 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_without_llm(self):
        """No LLM => returns unavailable message with 0.0 confidence."""
        pipeline = AgenticPipeline(None, None, None)
        answer, confidence = await pipeline._direct_answer("What is hypertension?")

        assert "unavailable" in answer.lower()
        assert confidence == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_returns_empty(self, mocker):
        """LLM returns empty string => fallback answer."""
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = ""
        pipeline = AgenticPipeline(None, mock_llm, None)
        answer, confidence = await pipeline._direct_answer("What is hypertension?")

        assert "Unable to generate" in answer or "unavailable" in answer.lower()
        assert confidence == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_returns_none(self, mocker):
        """LLM returns None => fallback answer."""
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = None
        pipeline = AgenticPipeline(None, mock_llm, None)
        answer, confidence = await pipeline._direct_answer("What is hypertension?")

        assert "Unable to generate" in answer or "unavailable" in answer.lower()
        assert confidence == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_exception_returns_fallback(self, mocker):
        """LLM raises exception => graceful fallback."""
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.side_effect = Exception("Ollama is down")
        pipeline = AgenticPipeline(None, mock_llm, None)
        answer, confidence = await pipeline._direct_answer("What is hypertension?")

        assert "Unable to generate" in answer
        assert confidence == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_answer_includes_disclaimer(self, mocker):
        """Direct answer includes the 'general medical knowledge' disclaimer."""
        mock_llm = mocker.AsyncMock()
        mock_llm.generate.return_value = (
            "Hypertension is high blood pressure.\nConfidence: 0.80"
        )
        pipeline = AgenticPipeline(None, mock_llm, None)
        answer, confidence = await pipeline._direct_answer("What is hypertension?")

        assert "general medical knowledge" in answer.lower()
        assert "clinical" in answer.lower() or "institutional" in answer.lower()
