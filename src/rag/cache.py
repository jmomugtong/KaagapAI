"""
MedQuery Cache Management Module

Two-tier caching for embeddings using Redis (fast) and PostgreSQL (persistent).
Implements 7-day TTL for Redis cache with content-addressable storage.
"""

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag.embedding import EmbeddingGenerator


# ============================================
# Constants
# ============================================

# Default TTL: 7 days in seconds
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60  # 604800 seconds

# Cache key prefix
CACHE_KEY_PREFIX = "emb:"


# ============================================
# Cache Manager
# ============================================


class CacheManager:
    """Manages embedding cache with Redis and optional PostgreSQL fallback.

    Provides a two-tier caching strategy:
    1. Redis: Fast in-memory cache with TTL (7 days default)
    2. PostgreSQL: Persistent storage via EmbeddingsCache model

    Attributes:
        ttl_seconds: Time-to-live for cached embeddings in seconds.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        """Initialize the cache manager.

        Args:
            redis_url: Redis connection URL. If None, uses REDIS_URL env var.
            ttl_seconds: Cache TTL in seconds. Default is 7 days.
        """
        self.ttl_seconds = ttl_seconds
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis: object | None = None

    async def _get_redis(self) -> object:
        """Lazily initialize Redis connection.

        Returns:
            Redis client instance.
        """
        if self._redis is None:
            import redis.asyncio as redis

            self._redis = redis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    def _make_key(self, text_hash: str) -> str:
        """Create cache key from text hash.

        Args:
            text_hash: SHA256 hash of the text content.

        Returns:
            Cache key with prefix.
        """
        return f"{CACHE_KEY_PREFIX}{text_hash}"

    async def get(self, text_hash: str) -> list[float] | None:
        """Get embedding from cache.

        Args:
            text_hash: SHA256 hash of the text content.

        Returns:
            Cached embedding vector if found, None otherwise.
        """
        try:
            redis = await self._get_redis()
            key = self._make_key(text_hash)
            cached = await redis.get(key)

            if cached:
                return json.loads(cached)
            return None
        except Exception:
            # Cache miss on error
            return None

    async def set(
        self,
        text_hash: str,
        embedding: list[float],
        ttl: int | None = None,
    ) -> bool:
        """Store embedding in cache.

        Args:
            text_hash: SHA256 hash of the text content.
            embedding: 384-dimensional embedding vector.
            ttl: Optional custom TTL in seconds.

        Returns:
            True if stored successfully, False otherwise.
        """
        try:
            redis = await self._get_redis()
            key = self._make_key(text_hash)
            ttl_seconds = ttl if ttl is not None else self.ttl_seconds

            await redis.set(key, json.dumps(embedding), ex=ttl_seconds)
            return True
        except Exception:
            return False

    async def delete(self, text_hash: str) -> bool:
        """Delete embedding from cache.

        Args:
            text_hash: SHA256 hash of the text content.

        Returns:
            True if deleted, False otherwise.
        """
        try:
            redis = await self._get_redis()
            key = self._make_key(text_hash)
            await redis.delete(key)
            return True
        except Exception:
            return False

    async def exists(self, text_hash: str) -> bool:
        """Check if embedding exists in cache.

        Args:
            text_hash: SHA256 hash of the text content.

        Returns:
            True if exists, False otherwise.
        """
        try:
            redis = await self._get_redis()
            key = self._make_key(text_hash)
            return bool(await redis.exists(key))
        except Exception:
            return False

    async def get_or_generate(
        self,
        text: str,
        generator: "EmbeddingGenerator",
    ) -> list[float]:
        """Get embedding from cache or generate if not cached.

        This is the primary interface for embedding retrieval.
        Checks cache first, generates and caches on miss.

        Args:
            text: Text to get embedding for.
            generator: EmbeddingGenerator instance for computing embeddings.

        Returns:
            384-dimensional embedding vector.
        """
        text_hash = generator.compute_hash(text)

        # Try cache first
        cached = await self.get(text_hash)
        if cached is not None:
            return cached

        # Generate embedding
        embedding = generator.generate(text)

        # Cache the result (fire and forget)
        await self.set(text_hash, embedding)

        return embedding

    async def get_or_generate_batch(
        self,
        texts: list[str],
        generator: "EmbeddingGenerator",
    ) -> list[list[float]]:
        """Get embeddings from cache or generate for batch of texts.

        Optimizes by checking cache first, then batch-generating
        only the missing embeddings.

        Args:
            texts: List of texts to get embeddings for.
            generator: EmbeddingGenerator instance.

        Returns:
            List of 384-dimensional embedding vectors.
        """
        if not texts:
            return []

        # Compute hashes
        hashes = [generator.compute_hash(text) for text in texts]

        # Check cache for all
        results: list[list[float] | None] = [None] * len(texts)
        texts_to_generate: list[tuple[int, str]] = []

        for idx, (text, text_hash) in enumerate(zip(texts, hashes, strict=False)):
            cached = await self.get(text_hash)
            if cached is not None:
                results[idx] = cached
            else:
                texts_to_generate.append((idx, text))

        # Generate missing embeddings in batch
        if texts_to_generate:
            indices = [t[0] for t in texts_to_generate]
            texts_batch = [t[1] for t in texts_to_generate]

            embeddings = generator.generate_batch(texts_batch)

            # Store results and cache
            for idx, _text, embedding in zip(indices, texts_batch, embeddings, strict=False):
                results[idx] = embedding
                text_hash = hashes[idx]
                await self.set(text_hash, embedding)

        # Type assertion: all results should be filled now
        return [r for r in results if r is not None]

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
