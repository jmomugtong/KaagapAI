import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.rag.cache import CacheManager
from src.rag.embedding import EmbeddingGenerator


@pytest.fixture
def mock_redis():
    """Create a mock async Redis instance."""
    mock = AsyncMock()
    mock.get.return_value = None  # Cache miss by default
    mock.close.return_value = None
    return mock


@pytest.fixture
def embedding_generator():
    """Create an EmbeddingGenerator with mocked model and cache."""
    with patch("src.rag.cache.Redis") as mock_redis_cls:
        # Create mock Redis instance
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = None
        mock_redis.close.return_value = None
        mock_redis_cls.from_url.return_value = mock_redis

        generator = EmbeddingGenerator(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Mock the actual transformer model to avoid loading heavy weights during test
        generator.model = MagicMock()
        generator.model.encode.return_value = np.random.rand(2, 384).astype(
            np.float32
        )  # 2 texts, 384 dims

        yield generator


@pytest.mark.asyncio
async def test_embedding_generation_batch(embedding_generator):
    """Test that embeddings are generated for batch of texts."""
    texts = ["Test sentence 1", "Test sentence 2"]
    embeddings = await embedding_generator.generate_embeddings(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert isinstance(embeddings[0], list)
    # Verify model.encode was called with correct batch
    embedding_generator.model.encode.assert_called_once()


@pytest.mark.asyncio
async def test_redis_cache_hit():
    """Test cache hit returns cached embedding."""
    with patch("src.rag.cache.Redis") as mock_redis_cls:
        # Setup mock Redis with cache hit
        mock_redis = AsyncMock()
        mock_redis.get.return_value = json.dumps([0.1, 0.2, 0.3])
        mock_redis_cls.from_url.return_value = mock_redis

        cache = CacheManager()
        result = await cache.get_embedding("test_hash")

        assert result == [0.1, 0.2, 0.3]
        mock_redis.get.assert_called_with("embedding:test_hash")


@pytest.mark.asyncio
async def test_redis_cache_miss_and_set():
    """Test cache miss returns None and set stores embedding."""
    with patch("src.rag.cache.Redis") as mock_redis_cls:
        # Setup mock Redis with cache miss
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = None
        mock_redis_cls.from_url.return_value = mock_redis

        cache = CacheManager()

        # Test cache miss
        result = await cache.get_embedding("test_hash")
        assert result is None

        # Test cache set
        embedding = [0.1, 0.2, 0.3]
        await cache.set_embedding("test_hash", embedding)

        # Verify setex was called with correct arguments
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]
        assert call_args[0] == "embedding:test_hash"
        assert call_args[1] == 604800  # Default TTL
        assert json.loads(call_args[2]) == embedding
