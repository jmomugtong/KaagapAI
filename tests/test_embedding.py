import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.cache import CacheManager
from src.rag.embedding import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    EmbeddingGenerator,
    LangChainEmbeddingAdapter,
)


@pytest.fixture
def mock_redis():
    """Create a mock async Redis instance."""
    mock = AsyncMock()
    mock.get.return_value = None  # Cache miss by default
    mock.close.return_value = None
    return mock


def _mock_ollama_embed_response(dim=768, count=1):
    """Build a mock httpx response for Ollama /api/embed."""
    embeddings = [[0.1] * dim for _ in range(count)]
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"embeddings": embeddings}
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


@pytest.fixture
def embedding_generator():
    """Create an EmbeddingGenerator with mocked cache."""
    with patch("src.rag.cache.Redis") as mock_redis_cls:
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = None
        mock_redis.close.return_value = None
        mock_redis_cls.from_url.return_value = mock_redis

        generator = EmbeddingGenerator(
            model_name="nomic-embed-text",
            ollama_url="http://fake:11434",
        )
        yield generator


@pytest.mark.asyncio
async def test_embedding_generation_batch(embedding_generator):
    """Test that embeddings are generated for batch of texts via Ollama."""
    texts = ["Test sentence 1", "Test sentence 2"]
    mock_resp = _mock_ollama_embed_response(dim=768, count=2)

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        embeddings = await embedding_generator.generate_embeddings(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768
    assert isinstance(embeddings[0], list)
    # Verify Ollama API was called
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_query_prefix_applied(embedding_generator):
    """Test that search_query prefix is applied when is_query=True."""
    texts = ["What is the treatment?"]
    mock_resp = _mock_ollama_embed_response(dim=768, count=1)

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await embedding_generator.generate_embeddings(texts, is_query=True)

    call_args = mock_client.post.call_args
    payload = call_args[1]["json"]
    assert payload["input"][0] == QUERY_PREFIX + "What is the treatment?"


@pytest.mark.asyncio
async def test_document_prefix_applied(embedding_generator):
    """Test that search_document prefix is applied for documents (is_query=False)."""
    texts = ["Document chunk text"]
    mock_resp = _mock_ollama_embed_response(dim=768, count=1)

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await embedding_generator.generate_embeddings(texts, is_query=False)

    call_args = mock_client.post.call_args
    payload = call_args[1]["json"]
    assert payload["input"][0] == DOCUMENT_PREFIX + "Document chunk text"


@pytest.mark.asyncio
async def test_ollama_model_passed_in_request(embedding_generator):
    """Test that the correct model name is passed to Ollama API."""
    mock_resp = _mock_ollama_embed_response(dim=768, count=1)

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await embedding_generator.generate_embeddings(["test"])

    call_args = mock_client.post.call_args
    payload = call_args[1]["json"]
    assert payload["model"] == "nomic-embed-text"


@pytest.mark.asyncio
async def test_redis_cache_hit():
    """Test cache hit returns cached embedding."""
    with patch("src.rag.cache.Redis") as mock_redis_cls:
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
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = None
        mock_redis_cls.from_url.return_value = mock_redis

        cache = CacheManager()

        result = await cache.get_embedding("test_hash")
        assert result is None

        embedding = [0.1, 0.2, 0.3]
        await cache.set_embedding("test_hash", embedding)

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]
        assert call_args[0] == "embedding:test_hash"
        assert call_args[1] == 604800  # Default TTL
        assert json.loads(call_args[2]) == embedding


class TestLangChainEmbeddingAdapter:
    """Tests for the LangChain adapter (sync Ollama calls)."""

    def test_embed_documents_uses_document_prefix(self):
        """embed_documents prepends search_document prefix."""
        with patch("src.rag.cache.Redis") as mock_redis_cls:
            mock_redis = AsyncMock()
            mock_redis_cls.from_url.return_value = mock_redis

            generator = EmbeddingGenerator(
                model_name="nomic-embed-text",
                ollama_url="http://fake:11434",
            )
            adapter = LangChainEmbeddingAdapter(generator)

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"embeddings": [[0.1] * 768]}
            mock_resp.raise_for_status = MagicMock()

            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)

            with patch("httpx.Client", return_value=mock_client):
                result = adapter.embed_documents(["Some text"])

            assert len(result) == 1
            assert len(result[0]) == 768
            payload = mock_client.post.call_args[1]["json"]
            assert payload["input"][0] == DOCUMENT_PREFIX + "Some text"

    def test_embed_query_uses_query_prefix(self):
        """embed_query prepends search_query prefix."""
        with patch("src.rag.cache.Redis") as mock_redis_cls:
            mock_redis = AsyncMock()
            mock_redis_cls.from_url.return_value = mock_redis

            generator = EmbeddingGenerator(
                model_name="nomic-embed-text",
                ollama_url="http://fake:11434",
            )
            adapter = LangChainEmbeddingAdapter(generator)

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"embeddings": [[0.2] * 768]}
            mock_resp.raise_for_status = MagicMock()

            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)

            with patch("httpx.Client", return_value=mock_client):
                result = adapter.embed_query("What is this?")

            assert len(result) == 768
            payload = mock_client.post.call_args[1]["json"]
            assert payload["input"][0] == QUERY_PREFIX + "What is this?"
