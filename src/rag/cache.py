"""
Cache Manager for MedQuery

Provides async Redis caching for embeddings and query responses.
Uses redis.asyncio for true async operations.
"""

import os
import json
from typing import Optional, List
from redis.asyncio import Redis


class CacheManager:
    """
    Async cache manager using Redis for embeddings and query responses.

    Features:
    - Two-tier caching: Redis (fast) + PostgreSQL (persistent)
    - Embedding cache: 7-day TTL by default
    - Query response cache: 1-hour TTL
    - LRU eviction policy
    """

    def __init__(self):
        self.redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._redis: Optional[Redis] = None
        self.ttl = int(os.environ.get("EMBEDDING_CACHE_TTL_SECONDS", 604800))

    async def _get_redis(self) -> Redis:
        """Get or create async Redis connection."""
        if self._redis is None:
            self._redis = Redis.from_url(
                self.redis_url, encoding="utf-8", decode_responses=True
            )
        return self._redis

    async def get_embedding(self, key: str) -> Optional[List[float]]:
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
            return json.loads(val)
        return None

    async def set_embedding(self, key: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.

        Args:
            key: SHA256 hash of the text
            embedding: 384-dimensional vector to cache
        """
        redis = await self._get_redis()
        await redis.setex(f"embedding:{key}", self.ttl, json.dumps(embedding))

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
