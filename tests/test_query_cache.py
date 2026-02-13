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
