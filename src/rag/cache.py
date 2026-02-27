"""
Cache Manager for KaagapAI

Provides async Redis caching for embeddings and query responses.
Uses redis.asyncio for true async operations.
"""

import hashlib
import json
import logging
import os

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Async cache manager using Redis for embeddings and query responses.

    Features:
    - Two-tier caching: Redis (fast) + PostgreSQL (persistent)
    - Embedding cache: 7-day TTL by default
    - Query response cache: 1-hour TTL
    - LRU eviction policy
    """

    def __init__(self) -> None:
        self.redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._redis: Redis | None = None
        self.ttl = int(os.environ.get("EMBEDDING_CACHE_TTL_SECONDS", 604800))
        self.query_ttl = int(os.environ.get("QUERY_CACHE_TTL_SECONDS", 3600))

    async def _get_redis(self) -> Redis:
        """Get or create async Redis connection."""
        if self._redis is None:
            self._redis = Redis.from_url(
                self.redis_url, encoding="utf-8", decode_responses=True
            )
        return self._redis

    async def get_embedding(self, key: str) -> list[float] | None:
        """
        Retrieve embedding from cache.

        Args:
            key: SHA256 hash of the text

        Returns:
            384-dimensional embedding vector or None if not cached
        """
        redis = await self._get_redis()
        val = await redis.get(f"embedding:{key}")
        if val:
            result: list[float] = json.loads(val)
            return result
        return None

    async def set_embedding(self, key: str, embedding: list[float]) -> None:
        """
        Store embedding in cache.

        Args:
            key: SHA256 hash of the text
            embedding: 384-dimensional vector to cache
        """
        redis = await self._get_redis()
        await redis.setex(f"embedding:{key}", self.ttl, json.dumps(embedding))

    async def get_query_result(self, query: str) -> dict | None:
        """
        Retrieve cached query result.

        Args:
            query: The user's query string (case-insensitive lookup).

        Returns:
            Cached response dict or None if not cached.
        """
        try:
            redis = await self._get_redis()
            key = self._query_cache_key(query)
            val = await redis.get(key)
            if val:
                cached: dict[str, object] = json.loads(val)
                return cached
        except Exception as e:
            logger.warning("Query cache read failed: %s", e)
        return None

    async def set_query_result(self, query: str, result: dict) -> None:
        """
        Store query result in cache.

        Args:
            query: The user's query string.
            result: Response dict to cache.
        """
        try:
            redis = await self._get_redis()
            key = self._query_cache_key(query)
            await redis.setex(key, self.query_ttl, json.dumps(result))
        except Exception as e:
            logger.warning("Query cache write failed: %s", e)

    @staticmethod
    def _query_cache_key(query: str) -> str:
        """Build a deterministic, case-insensitive cache key for a query."""
        normalized = query.lower().strip()
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        return f"query:{digest}"

    async def flush_embeddings(self) -> int:
        """Flush all cached embeddings. Returns count of keys deleted."""
        redis = await self._get_redis()
        keys = []
        async for key in redis.scan_iter("embedding:*"):
            keys.append(key)
        if keys:
            await redis.delete(*keys)
        return len(keys)

    async def flush_queries(self) -> int:
        """Flush all cached query results. Returns count of keys deleted."""
        redis = await self._get_redis()
        keys = []
        async for key in redis.scan_iter("query:*"):
            keys.append(key)
        if keys:
            await redis.delete(*keys)
        return len(keys)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
