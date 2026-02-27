"""
Tests for query-result caching (Phase 1 optimization).

Tests:
- Cache miss returns None
- Cache hit returns stored result
- Cache key is case-insensitive
- Cache failure does not break queries
"""

import pytest

from src.rag.cache import CacheManager


class TestQueryCacheKey:
    """Tests for deterministic, case-insensitive cache key generation."""

    @pytest.mark.unit
    def test_cache_key_deterministic(self):
        """Same query produces the same cache key."""
        key1 = CacheManager._query_cache_key("What is HTN?")
        key2 = CacheManager._query_cache_key("What is HTN?")
        assert key1 == key2

    @pytest.mark.unit
    def test_cache_key_case_insensitive(self):
        """'What is HTN?' and 'what is htn?' produce the same cache key."""
        key1 = CacheManager._query_cache_key("What is HTN?")
        key2 = CacheManager._query_cache_key("what is htn?")
        assert key1 == key2

    @pytest.mark.unit
    def test_cache_key_strips_whitespace(self):
        """Leading/trailing whitespace is ignored."""
        key1 = CacheManager._query_cache_key("What is HTN?")
        key2 = CacheManager._query_cache_key("  What is HTN?  ")
        assert key1 == key2

    @pytest.mark.unit
    def test_cache_key_has_query_prefix(self):
        """Cache key starts with 'query:' prefix."""
        key = CacheManager._query_cache_key("test query")
        assert key.startswith("query:")


class TestQueryCacheMissHit:
    """Tests for cache miss/hit behavior with a fake Redis."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, mocker):
        """get_query_result returns None when key is not in Redis."""
        cache = CacheManager()
        mock_redis = mocker.AsyncMock()
        mock_redis.get.return_value = None
        mocker.patch.object(cache, "_get_redis", return_value=mock_redis)

        result = await cache.get_query_result("non-existent query")
        assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_hit_returns_stored_result(self, mocker):
        """get_query_result returns the stored dict on a cache hit."""
        import json

        cache = CacheManager()
        stored = {"answer": "HTN is hypertension.", "confidence": 0.92}
        mock_redis = mocker.AsyncMock()
        mock_redis.get.return_value = json.dumps(stored)
        mocker.patch.object(cache, "_get_redis", return_value=mock_redis)

        result = await cache.get_query_result("What is HTN?")
        assert result == stored

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_set_query_result_stores_with_ttl(self, mocker):
        """set_query_result calls redis.setex with the correct TTL."""
        cache = CacheManager()
        mock_redis = mocker.AsyncMock()
        mocker.patch.object(cache, "_get_redis", return_value=mock_redis)

        data = {"answer": "test", "confidence": 0.9}
        await cache.set_query_result("test query", data)

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0].startswith("query:")
        assert call_args[0][1] == cache.query_ttl

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_failure_does_not_raise_on_get(self, mocker):
        """get_query_result returns None (not raises) when Redis errors."""
        cache = CacheManager()
        mock_redis = mocker.AsyncMock()
        mock_redis.get.side_effect = ConnectionError("Redis down")
        mocker.patch.object(cache, "_get_redis", return_value=mock_redis)

        result = await cache.get_query_result("test query")
        assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_failure_does_not_raise_on_set(self, mocker):
        """set_query_result silently swallows Redis errors."""
        cache = CacheManager()
        mock_redis = mocker.AsyncMock()
        mock_redis.setex.side_effect = ConnectionError("Redis down")
        mocker.patch.object(cache, "_get_redis", return_value=mock_redis)

        # Should not raise
        await cache.set_query_result("test query", {"answer": "x"})


class TestCacheFlushAndClose:
    """Tests for flush_embeddings, flush_queries, and close methods."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_flush_embeddings_returns_key_count(self, mocker):
        """flush_embeddings deletes all embedding: keys and returns count."""
        cache = CacheManager()
        mock_redis = mocker.AsyncMock()

        async def mock_scan_iter(pattern):
            for key in ["embedding:abc123", "embedding:def456"]:
                yield key

        mock_redis.scan_iter = mock_scan_iter
        mocker.patch.object(cache, "_get_redis", return_value=mock_redis)

        count = await cache.flush_embeddings()
        assert count == 2
        mock_redis.delete.assert_called_once_with(
            "embedding:abc123", "embedding:def456"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_flush_embeddings_empty_returns_zero(self, mocker):
        """flush_embeddings returns 0 and does not call delete when no keys exist."""
        cache = CacheManager()
        mock_redis = mocker.AsyncMock()

        async def mock_scan_iter(pattern):
            return
            yield  # make it an async generator

        mock_redis.scan_iter = mock_scan_iter
        mocker.patch.object(cache, "_get_redis", return_value=mock_redis)

        count = await cache.flush_embeddings()
        assert count == 0
        mock_redis.delete.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_flush_queries_returns_key_count(self, mocker):
        """flush_queries deletes all query: keys and returns count."""
        cache = CacheManager()
        mock_redis = mocker.AsyncMock()

        async def mock_scan_iter(pattern):
            for key in ["query:aaa", "query:bbb", "query:ccc"]:
                yield key

        mock_redis.scan_iter = mock_scan_iter
        mocker.patch.object(cache, "_get_redis", return_value=mock_redis)

        count = await cache.flush_queries()
        assert count == 3
        mock_redis.delete.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_flush_queries_empty_returns_zero(self, mocker):
        """flush_queries returns 0 when no query keys exist."""
        cache = CacheManager()
        mock_redis = mocker.AsyncMock()

        async def mock_scan_iter(pattern):
            return
            yield

        mock_redis.scan_iter = mock_scan_iter
        mocker.patch.object(cache, "_get_redis", return_value=mock_redis)

        count = await cache.flush_queries()
        assert count == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_closes_redis_and_clears_reference(self, mocker):
        """close() calls redis.close() and sets _redis to None."""
        cache = CacheManager()
        mock_redis = mocker.AsyncMock()
        cache._redis = mock_redis

        await cache.close()

        mock_redis.close.assert_called_once()
        assert cache._redis is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_when_no_connection_does_not_raise(self):
        """close() is a no-op when _redis is None."""
        cache = CacheManager()
        assert cache._redis is None
        await cache.close()  # Should not raise
