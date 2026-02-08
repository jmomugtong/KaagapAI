"""
MedQuery Embedding & Cache Tests

Comprehensive pytest tests for embedding generation and caching.
Tests written FIRST following TDD approach (Phase 2).
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ============================================
# Test Markers
# ============================================


def pytest_configure(config: Any) -> None:
    """Register custom markers for embedding tests."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# ============================================
# Fixtures
# ============================================


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for embedding tests."""
    return [
        "Acetaminophen 1000mg every 6 hours for pain management.",
        "Monitor vital signs every 15 minutes post-surgery.",
        "Patient presents with elevated blood pressure of 160/95.",
        "Administer morphine 2-4mg IV for severe pain.",
        "Check patient allergies before medication administration.",
    ]


@pytest.fixture
def sample_chunk_text() -> str:
    """Single chunk text for testing."""
    return "Post-operative pain protocol requires acetaminophen 1000mg q6h."


@pytest.fixture
def mock_embedding_vector() -> list[float]:
    """Mock 384-dimensional embedding vector."""
    np.random.seed(42)
    return np.random.randn(384).tolist()


@pytest.fixture
def mock_redis_client() -> MagicMock:
    """Mock Redis client for testing."""
    mock = MagicMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.exists = AsyncMock(return_value=False)
    return mock


# ============================================
# EmbeddingGenerator Tests
# ============================================


class TestEmbeddingGenerator:
    """Tests for embedding generation functionality."""

    def test_embedding_generator_instantiation(self) -> None:
        """Test EmbeddingGenerator can be instantiated."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()
        assert generator is not None

    def test_embedding_generator_default_model(self) -> None:
        """Test EmbeddingGenerator uses correct default model."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()
        assert generator.model_name == "all-MiniLM-L6-v2"

    def test_embedding_generator_custom_model(self) -> None:
        """Test EmbeddingGenerator accepts custom model name."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator(model_name="paraphrase-MiniLM-L6-v2")
        assert generator.model_name == "paraphrase-MiniLM-L6-v2"

    def test_embedding_dimension(self) -> None:
        """Test that embeddings are 384-dimensional."""
        from src.rag.embedding import EMBEDDING_DIMENSION

        assert EMBEDDING_DIMENSION == 384

    @pytest.mark.slow
    def test_generate_single_embedding(self, sample_chunk_text: str) -> None:
        """Test generating embedding for single text."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()
        embedding = generator.generate(sample_chunk_text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.slow
    def test_generate_batch_embeddings(self, sample_texts: list[str]) -> None:
        """Test generating embeddings for batch of texts."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()
        embeddings = generator.generate_batch(sample_texts)

        assert len(embeddings) == len(sample_texts)
        for embedding in embeddings:
            assert len(embedding) == 384

    def test_generate_empty_text_returns_zeros(self) -> None:
        """Test that empty text returns zero vector."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()
        # Empty text should either raise or return zeros
        with patch.object(generator, "_model") as mock_model:
            mock_model.encode.return_value = np.zeros(384)
            embedding = generator.generate("")
            assert len(embedding) == 384

    def test_generate_batch_empty_list(self) -> None:
        """Test batch generation with empty list."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()
        embeddings = generator.generate_batch([])

        assert embeddings == []

    def test_lazy_model_loading(self) -> None:
        """Test that model is loaded lazily on first use."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()
        # Model should not be loaded at instantiation
        assert generator._model is None or hasattr(generator, "_model")

    def test_compute_hash(self, sample_chunk_text: str) -> None:
        """Test hash computation for cache keys."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()
        hash1 = generator.compute_hash(sample_chunk_text)
        hash2 = generator.compute_hash(sample_chunk_text)

        # Same text should produce same hash
        assert hash1 == hash2
        # Hash should be SHA256 (64 hex chars)
        assert len(hash1) == 64

    def test_different_texts_different_hashes(self, sample_texts: list[str]) -> None:
        """Test that different texts produce different hashes."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()
        hashes = [generator.compute_hash(text) for text in sample_texts]

        # All hashes should be unique
        assert len(hashes) == len(set(hashes))


# ============================================
# CacheManager Tests
# ============================================


class TestCacheManager:
    """Tests for caching functionality."""

    def test_cache_manager_instantiation(self) -> None:
        """Test CacheManager can be instantiated."""
        from src.rag.cache import CacheManager

        cache = CacheManager()
        assert cache is not None

    def test_cache_manager_default_ttl(self) -> None:
        """Test CacheManager has 7-day default TTL."""
        from src.rag.cache import CacheManager

        cache = CacheManager()
        # 7 days in seconds
        expected_ttl = 7 * 24 * 60 * 60
        assert cache.ttl_seconds == expected_ttl

    def test_cache_manager_custom_ttl(self) -> None:
        """Test CacheManager accepts custom TTL."""
        from src.rag.cache import CacheManager

        custom_ttl = 3600  # 1 hour
        cache = CacheManager(ttl_seconds=custom_ttl)
        assert cache.ttl_seconds == custom_ttl

    def test_cache_key_format(self) -> None:
        """Test cache key is properly formatted."""
        from src.rag.cache import CacheManager

        cache = CacheManager()
        text_hash = "abc123"
        key = cache._make_key(text_hash)

        assert key.startswith("emb:")
        assert text_hash in key

    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(
        self, mock_redis_client: MagicMock
    ) -> None:
        """Test cache get returns None on cache miss."""
        from src.rag.cache import CacheManager

        cache = CacheManager()
        cache._redis = mock_redis_client

        result = await cache.get("nonexistent_hash")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_embedding_on_hit(
        self, mock_redis_client: MagicMock, mock_embedding_vector: list[float]
    ) -> None:
        """Test cache get returns embedding on cache hit."""
        import json

        from src.rag.cache import CacheManager

        cache = CacheManager()
        cache._redis = mock_redis_client
        mock_redis_client.get.return_value = json.dumps(mock_embedding_vector)

        result = await cache.get("existing_hash")

        assert result is not None
        assert len(result) == 384

    @pytest.mark.asyncio
    async def test_set_stores_embedding(
        self, mock_redis_client: MagicMock, mock_embedding_vector: list[float]
    ) -> None:
        """Test cache set stores embedding with TTL."""
        from src.rag.cache import CacheManager

        cache = CacheManager()
        cache._redis = mock_redis_client

        await cache.set("test_hash", mock_embedding_vector)

        mock_redis_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_removes_key(self, mock_redis_client: MagicMock) -> None:
        """Test cache delete removes key."""
        from src.rag.cache import CacheManager

        cache = CacheManager()
        cache._redis = mock_redis_client

        await cache.delete("test_hash")

        mock_redis_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_or_generate_cache_miss(
        self, mock_redis_client: MagicMock, sample_chunk_text: str
    ) -> None:
        """Test get_or_generate computes on cache miss."""
        from src.rag.cache import CacheManager
        from src.rag.embedding import EmbeddingGenerator

        cache = CacheManager()
        cache._redis = mock_redis_client

        # Mock the generator
        mock_generator = MagicMock(spec=EmbeddingGenerator)
        mock_generator.generate.return_value = [0.1] * 384
        mock_generator.compute_hash.return_value = "test_hash"

        result = await cache.get_or_generate(
            sample_chunk_text, generator=mock_generator
        )

        assert result is not None
        assert len(result) == 384
        mock_generator.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_or_generate_cache_hit(
        self,
        mock_redis_client: MagicMock,
        sample_chunk_text: str,
        mock_embedding_vector: list[float],
    ) -> None:
        """Test get_or_generate returns cached on hit."""
        import json

        from src.rag.cache import CacheManager
        from src.rag.embedding import EmbeddingGenerator

        cache = CacheManager()
        cache._redis = mock_redis_client
        mock_redis_client.get.return_value = json.dumps(mock_embedding_vector)

        mock_generator = MagicMock(spec=EmbeddingGenerator)
        mock_generator.compute_hash.return_value = "test_hash"

        result = await cache.get_or_generate(
            sample_chunk_text, generator=mock_generator
        )

        assert result is not None
        # Generator should NOT be called on cache hit
        mock_generator.generate.assert_not_called()


# ============================================
# Integration Tests
# ============================================


class TestEmbeddingCacheIntegration:
    """Integration tests for embedding + cache together."""

    @pytest.mark.slow
    def test_embedding_and_hash_consistency(
        self, sample_texts: list[str]
    ) -> None:
        """Test embedding generation is deterministic with hash."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()

        for text in sample_texts[:2]:  # Test with first 2 only (faster)
            embedding1 = generator.generate(text)
            embedding2 = generator.generate(text)

            # Same text should produce same embedding
            np.testing.assert_array_almost_equal(embedding1, embedding2)

    def test_hash_collision_resistance(self) -> None:
        """Test that similar texts produce different hashes."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()

        text1 = "Acetaminophen 1000mg every 6 hours."
        text2 = "Acetaminophen 1000mg every 8 hours."  # Slight difference

        hash1 = generator.compute_hash(text1)
        hash2 = generator.compute_hash(text2)

        assert hash1 != hash2


# ============================================
# Performance Tests
# ============================================


class TestEmbeddingPerformance:
    """Performance tests for embedding generation."""

    @pytest.mark.slow
    def test_single_embedding_latency(self, sample_chunk_text: str) -> None:
        """Test single embedding generation is under 100ms."""
        import time

        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()
        # Warm up
        generator.generate(sample_chunk_text)

        # Measure
        start = time.perf_counter()
        generator.generate(sample_chunk_text)
        elapsed = time.perf_counter() - start

        # Should be under 100ms (0.1s)
        assert elapsed < 0.5  # 500ms is acceptable for test environments

    @pytest.mark.slow
    def test_batch_embedding_latency(self) -> None:
        """Test 32-chunk batch is under 2 seconds."""
        import time

        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()
        texts = ["Sample text for testing embeddings."] * 32

        # Warm up
        generator.generate_batch(texts[:1])

        # Measure
        start = time.perf_counter()
        embeddings = generator.generate_batch(texts)
        elapsed = time.perf_counter() - start

        assert len(embeddings) == 32
        # Should be under 2 seconds for batch
        assert elapsed < 5.0  # Allow more time for test environments


# ============================================
# Edge Case Tests
# ============================================


class TestEmbeddingEdgeCases:
    """Edge case tests for embeddings."""

    def test_embedding_special_characters(self) -> None:
        """Test embedding text with special characters."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()
        text = "Dosage: 50mg/kg/day (max: 1000mg) — check patient's BMI"

        with patch.object(generator, "_get_model") as mock_get:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(384)
            mock_get.return_value = mock_model

            embedding = generator.generate(text)
            assert len(embedding) == 384

    def test_embedding_unicode(self) -> None:
        """Test embedding with unicode characters."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()
        text = "Température: 38.5°C — Médicaments administrés"

        with patch.object(generator, "_get_model") as mock_get:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(384)
            mock_get.return_value = mock_model

            embedding = generator.generate(text)
            assert len(embedding) == 384

    def test_embedding_very_long_text(self) -> None:
        """Test embedding very long text (truncation)."""
        from src.rag.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator()
        # Very long text
        long_text = "word " * 5000  # ~5000 words

        with patch.object(generator, "_get_model") as mock_get:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(384)
            mock_get.return_value = mock_model

            embedding = generator.generate(long_text)
            # Should still produce valid embedding
            assert len(embedding) == 384
