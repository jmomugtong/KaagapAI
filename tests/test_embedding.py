import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.cache import CacheManager
from src.rag.embedding import (
    DOCUMENT_PREFIX,
    EMBEDDING_BATCH_SIZE,
    QUERY_PREFIX,
    EmbeddingGenerator,
    LangChainEmbeddingAdapter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embeddings(dim: int = 768, count: int = 1) -> list[list[float]]:
    """Return a list of fake embedding vectors."""
    return [[0.1] * dim for _ in range(count)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_st_model():
    """Fake SentenceTransformer-like model that does no real work."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 768
    return model


@pytest.fixture
def embedding_generator():
    """Create an EmbeddingGenerator with a mocked Redis cache."""
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


# ---------------------------------------------------------------------------
# EmbeddingGenerator — async tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_generation_batch(embedding_generator, mock_st_model):
    """Embeddings are generated for a batch of texts via sentence-transformers."""
    texts = ["Test sentence 1", "Test sentence 2"]
    fake_embs = _fake_embeddings(count=2)

    with patch("src.rag.embedding._load_st_model", return_value=mock_st_model):
        with patch("src.rag.embedding._encode_sync", return_value=fake_embs):
            embeddings = await embedding_generator.generate_embeddings(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768
    assert isinstance(embeddings[0], list)


@pytest.mark.asyncio
async def test_query_prefix_applied(embedding_generator, mock_st_model):
    """search_query prefix is prepended when is_query=True."""
    texts = ["What is the treatment?"]
    fake_embs = _fake_embeddings(count=1)

    with patch("src.rag.embedding._load_st_model", return_value=mock_st_model):
        with patch(
            "src.rag.embedding._encode_sync", return_value=fake_embs
        ) as mock_encode:
            await embedding_generator.generate_embeddings(texts, is_query=True)

    # _encode_sync(model, prefixed_texts, batch_size) — second positional arg
    prefixed_texts = mock_encode.call_args[0][1]
    assert prefixed_texts[0] == QUERY_PREFIX + "What is the treatment?"


@pytest.mark.asyncio
async def test_document_prefix_applied(embedding_generator, mock_st_model):
    """search_document prefix is prepended when is_query=False."""
    texts = ["Document chunk text"]
    fake_embs = _fake_embeddings(count=1)

    with patch("src.rag.embedding._load_st_model", return_value=mock_st_model):
        with patch(
            "src.rag.embedding._encode_sync", return_value=fake_embs
        ) as mock_encode:
            await embedding_generator.generate_embeddings(texts, is_query=False)

    prefixed_texts = mock_encode.call_args[0][1]
    assert prefixed_texts[0] == DOCUMENT_PREFIX + "Document chunk text"


@pytest.mark.asyncio
async def test_model_name_passed_to_loader(embedding_generator, mock_st_model):
    """_load_st_model is called with the configured model name."""
    fake_embs = _fake_embeddings(count=1)

    with patch(
        "src.rag.embedding._load_st_model", return_value=mock_st_model
    ) as mock_load:
        with patch("src.rag.embedding._encode_sync", return_value=fake_embs):
            await embedding_generator.generate_embeddings(["test"])

    mock_load.assert_called_with("nomic-embed-text")


@pytest.mark.asyncio
async def test_cache_not_called_when_disabled(embedding_generator, mock_st_model):
    """No cache writes when cache=False."""
    fake_embs = _fake_embeddings(count=1)

    with patch("src.rag.embedding._load_st_model", return_value=mock_st_model):
        with patch("src.rag.embedding._encode_sync", return_value=fake_embs):
            with patch.object(embedding_generator.cache, "set_embedding") as mock_set:
                await embedding_generator.generate_embeddings(["text"], cache=False)

    mock_set.assert_not_called()


@pytest.mark.asyncio
async def test_cache_written_when_enabled(embedding_generator, mock_st_model):
    """Embedding is written to cache when cache=True."""
    fake_embs = _fake_embeddings(count=1)

    with patch("src.rag.embedding._load_st_model", return_value=mock_st_model):
        with patch("src.rag.embedding._encode_sync", return_value=fake_embs):
            with patch.object(embedding_generator.cache, "set_embedding") as mock_set:
                await embedding_generator.generate_embeddings(["text"], cache=True)

    mock_set.assert_called_once()


# ---------------------------------------------------------------------------
# CacheManager — Redis tests (unchanged behaviour)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redis_cache_hit():
    """Cache hit returns the stored embedding."""
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
    """Cache miss returns None; set stores the embedding with correct TTL."""
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
        assert call_args[1] == 604800  # Default TTL (7 days)
        assert json.loads(call_args[2]) == embedding


# ---------------------------------------------------------------------------
# LangChainEmbeddingAdapter — sync tests
# ---------------------------------------------------------------------------


class TestLangChainEmbeddingAdapter:
    """Tests for the LangChain adapter (synchronous calls)."""

    def _make_adapter(self, model_name: str = "nomic-embed-text"):
        with patch("src.rag.cache.Redis") as mock_redis_cls:
            mock_redis = AsyncMock()
            mock_redis_cls.from_url.return_value = mock_redis
            generator = EmbeddingGenerator(model_name=model_name)
            return LangChainEmbeddingAdapter(generator)

    def test_embed_documents_uses_document_prefix(self):
        """embed_documents prepends search_document prefix."""
        mock_model = MagicMock()
        adapter = self._make_adapter()
        fake_embs = _fake_embeddings(count=1)

        with patch("src.rag.embedding._load_st_model", return_value=mock_model):
            with patch(
                "src.rag.embedding._encode_sync", return_value=fake_embs
            ) as mock_encode:
                result = adapter.embed_documents(["Some text"])

        assert len(result) == 1
        assert len(result[0]) == 768
        prefixed_texts = mock_encode.call_args[0][1]
        assert prefixed_texts[0] == DOCUMENT_PREFIX + "Some text"

    def test_embed_query_uses_query_prefix(self):
        """embed_query prepends search_query prefix."""
        mock_model = MagicMock()
        adapter = self._make_adapter()
        fake_embs = _fake_embeddings(count=1)

        with patch("src.rag.embedding._load_st_model", return_value=mock_model):
            with patch(
                "src.rag.embedding._encode_sync", return_value=fake_embs
            ) as mock_encode:
                result = adapter.embed_query("What is this?")

        assert len(result) == 768
        prefixed_texts = mock_encode.call_args[0][1]
        assert prefixed_texts[0] == QUERY_PREFIX + "What is this?"

    def test_embed_documents_batch_size_passed(self):
        """embed_documents passes EMBEDDING_BATCH_SIZE to _encode_sync."""
        mock_model = MagicMock()
        adapter = self._make_adapter()
        fake_embs = _fake_embeddings(count=2)

        with patch("src.rag.embedding._load_st_model", return_value=mock_model):
            with patch(
                "src.rag.embedding._encode_sync", return_value=fake_embs
            ) as mock_encode:
                adapter.embed_documents(["Text A", "Text B"])

        batch_size_arg = mock_encode.call_args[0][2]
        assert batch_size_arg == EMBEDDING_BATCH_SIZE
