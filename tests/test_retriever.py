"""
Tests for Phase 4: Hybrid Retrieval System

Tests cover:
- QueryPreprocessor (abbreviation expansion, stop words, normalization)
- BM25Retriever (keyword search, scoring, normalization)
- VectorRetriever (cosine similarity via pgvector)
- HybridRetriever (fusion, deduplication, top-k, fallback)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.retriever import (
    BM25Retriever,
    HybridRetriever,
    QueryPreprocessor,
    ScoredChunk,
    VectorRetriever,
)

# ============================================
# Sample Data Fixtures
# ============================================


@pytest.fixture
def sample_chunks():
    """Create sample DocumentChunk-like objects for testing."""
    chunks = []
    texts = [
        "Administer acetaminophen 1000mg every 6 hours for post-operative pain management.",
        "Myocardial infarction patients should receive aspirin 325mg immediately upon arrival.",
        "Deep vein thrombosis prevention includes low-molecular-weight heparin administration.",
        "Congestive heart failure management involves ACE inhibitors and diuretics.",
        "Post-operative knee replacement requires early mobilization within 24 hours.",
        "Chronic obstructive pulmonary disease exacerbation treated with bronchodilators.",
        "Hypertension management with first-line agents includes ACE inhibitors and ARBs.",
        "Diabetes mellitus type 2 management includes metformin as first-line therapy.",
        "Pulmonary embolism treatment with anticoagulation therapy using heparin.",
        "Pain management protocol recommends multimodal analgesia for surgical patients.",
    ]
    for i, text in enumerate(texts):
        chunk = MagicMock()
        chunk.id = i + 1
        chunk.content = text
        chunk.document_id = (i // 3) + 1  # Group 3 chunks per document
        chunk.chunk_index = i
        chunk.embedding = [0.1 * (i + 1)] * 384  # Dummy embedding
        chunks.append(chunk)
    return chunks


@pytest.fixture
def preprocessor():
    return QueryPreprocessor()


@pytest.fixture
def bm25_retriever(sample_chunks):
    return BM25Retriever(sample_chunks)


# ============================================
# QueryPreprocessor Tests
# ============================================


class TestQueryPreprocessor:
    """Tests for query preprocessing."""

    def test_lowercase_normalization(self, preprocessor):
        """Query text is lowercased."""
        result = preprocessor.preprocess("What Is The Pain Protocol?")
        assert result == result.lower()

    def test_medical_abbreviation_expansion_mi(self, preprocessor):
        """MI expands to myocardial infarction."""
        result = preprocessor.preprocess("treatment for MI")
        assert "myocardial infarction" in result

    def test_medical_abbreviation_expansion_chf(self, preprocessor):
        """CHF expands to congestive heart failure."""
        result = preprocessor.preprocess("management of CHF")
        assert "congestive heart failure" in result

    def test_medical_abbreviation_expansion_dvt(self, preprocessor):
        """DVT expands to deep vein thrombosis."""
        result = preprocessor.preprocess("prevention of DVT")
        assert "deep vein thrombosis" in result

    def test_medical_abbreviation_expansion_pe(self, preprocessor):
        """PE expands to pulmonary embolism."""
        result = preprocessor.preprocess("diagnosis of PE")
        assert "pulmonary embolism" in result

    def test_medical_abbreviation_expansion_copd(self, preprocessor):
        """COPD expands to chronic obstructive pulmonary disease."""
        result = preprocessor.preprocess("treatment for COPD")
        assert "chronic obstructive pulmonary disease" in result

    def test_medical_abbreviation_expansion_htn(self, preprocessor):
        """HTN expands to hypertension."""
        result = preprocessor.preprocess("managing HTN")
        assert "hypertension" in result

    def test_medical_abbreviation_expansion_dm(self, preprocessor):
        """DM expands to diabetes mellitus."""
        result = preprocessor.preprocess("DM type 2 management")
        assert "diabetes mellitus" in result

    def test_stop_word_removal(self, preprocessor):
        """Common stop words are removed from tokenized output."""
        tokens = preprocessor.tokenize("what is the protocol for pain")
        assert "what" not in tokens
        assert "is" not in tokens
        assert "the" not in tokens
        assert "for" not in tokens
        assert "protocol" in tokens
        assert "pain" in tokens

    def test_tokenize_returns_list(self, preprocessor):
        """Tokenize returns a list of strings."""
        tokens = preprocessor.tokenize("pain management protocol")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_preprocess_preserves_medical_terms(self, preprocessor):
        """Medical terms not in abbreviation dict are preserved."""
        result = preprocessor.preprocess("acetaminophen dosage")
        assert "acetaminophen" in result


# ============================================
# BM25Retriever Tests
# ============================================


class TestBM25Retriever:
    """Tests for BM25 keyword search."""

    def test_search_returns_scored_chunks(self, bm25_retriever):
        """Search returns list of ScoredChunk objects."""
        results = bm25_retriever.search("pain management", top_k=5)
        assert all(isinstance(r, ScoredChunk) for r in results)

    def test_search_respects_top_k(self, bm25_retriever):
        """Search returns at most top_k results."""
        results = bm25_retriever.search("pain management", top_k=3)
        assert len(results) <= 3

    def test_search_default_top_k_10(self, bm25_retriever):
        """Default top_k is 10."""
        results = bm25_retriever.search("pain management protocol")
        assert len(results) <= 10

    def test_search_scores_normalized(self, bm25_retriever):
        """BM25 scores are normalized to 0.0-1.0 range."""
        results = bm25_retriever.search("pain management")
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_search_source_is_bm25(self, bm25_retriever):
        """Source field is 'bm25' for BM25 results."""
        results = bm25_retriever.search("acetaminophen")
        for r in results:
            assert r.source == "bm25"

    def test_search_relevant_results_ranked_higher(self, bm25_retriever):
        """More relevant results have higher scores."""
        results = bm25_retriever.search("acetaminophen pain")
        if len(results) >= 2:
            # Scores should be in descending order
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score

    def test_search_no_results(self, bm25_retriever):
        """Search returns empty list for unrelated query."""
        results = bm25_retriever.search("xyznonexistentterm")
        assert results == []

    def test_search_chunk_fields_populated(self, bm25_retriever):
        """ScoredChunk has all required fields populated."""
        results = bm25_retriever.search("pain management")
        if results:
            chunk = results[0]
            assert chunk.chunk_id is not None
            assert chunk.content is not None
            assert chunk.document_id is not None
            assert chunk.chunk_index is not None
            assert chunk.score is not None

    def test_empty_corpus(self):
        """BM25Retriever handles empty chunk list."""
        retriever = BM25Retriever([])
        results = retriever.search("pain")
        assert results == []

    def test_build_index_uses_preprocessor(self, sample_chunks):
        """BM25 index uses preprocessed (tokenized) text."""
        retriever = BM25Retriever(sample_chunks)
        # The retriever should have built an index
        assert retriever._index is not None


# ============================================
# VectorRetriever Tests
# ============================================


class TestVectorRetriever:
    """Tests for vector similarity search using pgvector."""

    @pytest.mark.asyncio
    async def test_search_returns_scored_chunks(self):
        """Vector search returns ScoredChunk objects."""
        mock_session = AsyncMock()
        # Mock the execute result
        mock_row = MagicMock()
        mock_row.id = 1
        mock_row.chunk_text = "Acetaminophen 1000mg for pain."
        mock_row.document_id = 1
        mock_row.chunk_index = 0
        mock_row.similarity = 0.92
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result

        retriever = VectorRetriever(mock_session)
        query_embedding = [0.1] * 384
        results = await retriever.search(query_embedding, top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], ScoredChunk)
        assert results[0].score == 0.92
        assert results[0].source == "vector"

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self):
        """Vector search passes top_k to the query."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        retriever = VectorRetriever(mock_session)
        query_embedding = [0.1] * 384
        await retriever.search(query_embedding, top_k=3)

        # Verify execute was called (the SQL should contain LIMIT)
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_source_is_vector(self):
        """Source field is 'vector' for vector results."""
        mock_session = AsyncMock()
        mock_row = MagicMock()
        mock_row.id = 1
        mock_row.chunk_text = "Some text"
        mock_row.document_id = 1
        mock_row.chunk_index = 0
        mock_row.similarity = 0.85
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result

        retriever = VectorRetriever(mock_session)
        results = await retriever.search([0.1] * 384)
        assert all(r.source == "vector" for r in results)

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        """Vector search returns empty list when no matches."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        retriever = VectorRetriever(mock_session)
        results = await retriever.search([0.1] * 384)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_handles_db_error(self):
        """Vector search returns empty list on database error."""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("DB connection failed")

        retriever = VectorRetriever(mock_session)
        results = await retriever.search([0.1] * 384)
        assert results == []


# ============================================
# HybridRetriever Tests
# ============================================


class TestHybridRetriever:
    """Tests for hybrid retrieval with fusion scoring."""

    @pytest.mark.asyncio
    async def test_hybrid_returns_scored_chunks(self, sample_chunks):
        """Hybrid search returns ScoredChunk objects."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        retriever = HybridRetriever(
            chunks=sample_chunks,
            session=mock_session,
        )
        results = await retriever.search(
            query="pain management",
            query_embedding=[0.1] * 384,
            top_k=5,
        )
        assert all(isinstance(r, ScoredChunk) for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_default_top_k_5(self, sample_chunks):
        """Default top_k for hybrid search is 5."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        retriever = HybridRetriever(
            chunks=sample_chunks,
            session=mock_session,
        )
        results = await retriever.search(
            query="pain management protocol acetaminophen",
            query_embedding=[0.1] * 384,
        )
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_hybrid_fusion_scoring(self, sample_chunks):
        """Fusion score = 0.4 * bm25 + 0.6 * cosine."""
        # Create a hybrid retriever with mocked sub-retrievers
        mock_session = AsyncMock()

        retriever = HybridRetriever(
            chunks=sample_chunks,
            session=mock_session,
        )

        bm25_result = ScoredChunk(
            chunk_id=1,
            content="pain management text",
            document_id=1,
            chunk_index=0,
            score=0.8,
            source="bm25",
        )
        vector_result = ScoredChunk(
            chunk_id=1,
            content="pain management text",
            document_id=1,
            chunk_index=0,
            score=0.9,
            source="vector",
        )

        with (
            patch.object(retriever._bm25, "search", return_value=[bm25_result]),
            patch.object(
                retriever._vector,
                "search",
                new_callable=AsyncMock,
                return_value=[vector_result],
            ),
        ):
            results = await retriever.search(
                query="pain management",
                query_embedding=[0.1] * 384,
                top_k=5,
            )

        assert len(results) == 1
        expected_score = 0.4 * 0.8 + 0.6 * 0.9  # 0.86
        assert abs(results[0].score - expected_score) < 1e-6
        assert results[0].source == "hybrid"

    @pytest.mark.asyncio
    async def test_hybrid_deduplication(self, sample_chunks):
        """Same chunk from both sources appears once with combined score."""
        mock_session = AsyncMock()

        retriever = HybridRetriever(
            chunks=sample_chunks,
            session=mock_session,
        )

        # Same chunk appears in both BM25 and vector results
        bm25_results = [
            ScoredChunk(
                chunk_id=1,
                content="text",
                document_id=1,
                chunk_index=0,
                score=0.7,
                source="bm25",
            ),
            ScoredChunk(
                chunk_id=2,
                content="text2",
                document_id=1,
                chunk_index=1,
                score=0.5,
                source="bm25",
            ),
        ]
        vector_results = [
            ScoredChunk(
                chunk_id=1,
                content="text",
                document_id=1,
                chunk_index=0,
                score=0.9,
                source="vector",
            ),
            ScoredChunk(
                chunk_id=3,
                content="text3",
                document_id=2,
                chunk_index=0,
                score=0.8,
                source="vector",
            ),
        ]

        with (
            patch.object(retriever._bm25, "search", return_value=bm25_results),
            patch.object(
                retriever._vector,
                "search",
                new_callable=AsyncMock,
                return_value=vector_results,
            ),
        ):
            results = await retriever.search(
                query="test",
                query_embedding=[0.1] * 384,
                top_k=5,
            )

        # chunk_id=1 appears in both, should be merged (not duplicated)
        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found"
        assert 1 in chunk_ids
        assert 2 in chunk_ids
        assert 3 in chunk_ids

    @pytest.mark.asyncio
    async def test_hybrid_dedup_combined_score(self, sample_chunks):
        """Deduplicated chunk gets fusion score from both sources."""
        mock_session = AsyncMock()

        retriever = HybridRetriever(
            chunks=sample_chunks,
            session=mock_session,
        )

        bm25_results = [
            ScoredChunk(
                chunk_id=1,
                content="text",
                document_id=1,
                chunk_index=0,
                score=0.8,
                source="bm25",
            ),
        ]
        vector_results = [
            ScoredChunk(
                chunk_id=1,
                content="text",
                document_id=1,
                chunk_index=0,
                score=0.9,
                source="vector",
            ),
        ]

        with (
            patch.object(retriever._bm25, "search", return_value=bm25_results),
            patch.object(
                retriever._vector,
                "search",
                new_callable=AsyncMock,
                return_value=vector_results,
            ),
        ):
            results = await retriever.search(
                query="test",
                query_embedding=[0.1] * 384,
                top_k=5,
            )

        chunk_1 = next(r for r in results if r.chunk_id == 1)
        expected = 0.4 * 0.8 + 0.6 * 0.9
        assert abs(chunk_1.score - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_hybrid_bm25_only_chunk_score(self, sample_chunks):
        """Chunk only in BM25 gets score = 0.4 * bm25_score."""
        mock_session = AsyncMock()

        retriever = HybridRetriever(
            chunks=sample_chunks,
            session=mock_session,
        )

        bm25_results = [
            ScoredChunk(
                chunk_id=2,
                content="text2",
                document_id=1,
                chunk_index=1,
                score=0.6,
                source="bm25",
            ),
        ]
        vector_results = []

        with (
            patch.object(retriever._bm25, "search", return_value=bm25_results),
            patch.object(
                retriever._vector,
                "search",
                new_callable=AsyncMock,
                return_value=vector_results,
            ),
        ):
            results = await retriever.search(
                query="test",
                query_embedding=[0.1] * 384,
                top_k=5,
            )

        chunk_2 = next(r for r in results if r.chunk_id == 2)
        expected = 0.4 * 0.6
        assert abs(chunk_2.score - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_hybrid_vector_only_chunk_score(self, sample_chunks):
        """Chunk only in vector gets score = 0.6 * cosine_score."""
        mock_session = AsyncMock()

        retriever = HybridRetriever(
            chunks=sample_chunks,
            session=mock_session,
        )

        bm25_results = []
        vector_results = [
            ScoredChunk(
                chunk_id=3,
                content="text3",
                document_id=2,
                chunk_index=0,
                score=0.85,
                source="vector",
            ),
        ]

        with (
            patch.object(retriever._bm25, "search", return_value=bm25_results),
            patch.object(
                retriever._vector,
                "search",
                new_callable=AsyncMock,
                return_value=vector_results,
            ),
        ):
            results = await retriever.search(
                query="test",
                query_embedding=[0.1] * 384,
                top_k=5,
            )

        chunk_3 = next(r for r in results if r.chunk_id == 3)
        expected = 0.6 * 0.85
        assert abs(chunk_3.score - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_hybrid_results_sorted_by_score(self, sample_chunks):
        """Results are sorted by descending fusion score."""
        mock_session = AsyncMock()

        retriever = HybridRetriever(
            chunks=sample_chunks,
            session=mock_session,
        )

        bm25_results = [
            ScoredChunk(
                chunk_id=1,
                content="t1",
                document_id=1,
                chunk_index=0,
                score=0.9,
                source="bm25",
            ),
            ScoredChunk(
                chunk_id=2,
                content="t2",
                document_id=1,
                chunk_index=1,
                score=0.3,
                source="bm25",
            ),
        ]
        vector_results = [
            ScoredChunk(
                chunk_id=3,
                content="t3",
                document_id=2,
                chunk_index=0,
                score=0.95,
                source="vector",
            ),
            ScoredChunk(
                chunk_id=1,
                content="t1",
                document_id=1,
                chunk_index=0,
                score=0.5,
                source="vector",
            ),
        ]

        with (
            patch.object(retriever._bm25, "search", return_value=bm25_results),
            patch.object(
                retriever._vector,
                "search",
                new_callable=AsyncMock,
                return_value=vector_results,
            ),
        ):
            results = await retriever.search(
                query="test",
                query_embedding=[0.1] * 384,
                top_k=5,
            )

        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    @pytest.mark.asyncio
    async def test_hybrid_fallback_bm25_only(self, sample_chunks):
        """Falls back to BM25-only when vector search fails."""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("pgvector unavailable")

        retriever = HybridRetriever(
            chunks=sample_chunks,
            session=mock_session,
        )
        results = await retriever.search(
            query="pain management",
            query_embedding=[0.1] * 384,
            top_k=5,
        )

        # Should still return results from BM25
        assert len(results) > 0
        # BM25-only results still get weighted scoring (0.4 * bm25)
        for r in results:
            assert r.score > 0

    @pytest.mark.asyncio
    async def test_hybrid_empty_results(self):
        """Returns empty list when no chunks match from either source."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        retriever = HybridRetriever(
            chunks=[],
            session=mock_session,
        )
        results = await retriever.search(
            query="xyznonexistent",
            query_embedding=[0.1] * 384,
            top_k=5,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_hybrid_top_k_limits_results(self, sample_chunks):
        """Top-k parameter limits the number of returned results."""
        mock_session = AsyncMock()

        retriever = HybridRetriever(
            chunks=sample_chunks,
            session=mock_session,
        )

        # Create many results from both sources
        bm25_results = [
            ScoredChunk(
                chunk_id=i,
                content=f"text{i}",
                document_id=1,
                chunk_index=i,
                score=0.5 + i * 0.05,
                source="bm25",
            )
            for i in range(1, 8)
        ]
        vector_results = [
            ScoredChunk(
                chunk_id=i,
                content=f"text{i}",
                document_id=1,
                chunk_index=i,
                score=0.6 + i * 0.03,
                source="vector",
            )
            for i in range(1, 8)
        ]

        with (
            patch.object(retriever._bm25, "search", return_value=bm25_results),
            patch.object(
                retriever._vector,
                "search",
                new_callable=AsyncMock,
                return_value=vector_results,
            ),
        ):
            results = await retriever.search(
                query="test",
                query_embedding=[0.1] * 384,
                top_k=5,
            )

        assert len(results) == 5


# ============================================
# ScoredChunk Tests
# ============================================


class TestScoredChunk:
    """Tests for the ScoredChunk dataclass."""

    def test_scored_chunk_creation(self):
        """ScoredChunk can be created with all fields."""
        chunk = ScoredChunk(
            chunk_id=1,
            content="Some clinical text.",
            document_id=1,
            chunk_index=0,
            score=0.95,
            source="hybrid",
        )
        assert chunk.chunk_id == 1
        assert chunk.content == "Some clinical text."
        assert chunk.document_id == 1
        assert chunk.chunk_index == 0
        assert chunk.score == 0.95
        assert chunk.source == "hybrid"

    def test_scored_chunk_equality_by_values(self):
        """Two ScoredChunks with same values are equal."""
        c1 = ScoredChunk(
            chunk_id=1,
            content="text",
            document_id=1,
            chunk_index=0,
            score=0.9,
            source="bm25",
        )
        c2 = ScoredChunk(
            chunk_id=1,
            content="text",
            document_id=1,
            chunk_index=0,
            score=0.9,
            source="bm25",
        )
        assert c1 == c2
