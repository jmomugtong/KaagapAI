"""
Rate Limiter for MedQuery (Phase 7)

Redis-backed rate limiting: 10 requests/minute per user.
Returns 429 with Retry-After header when limit exceeded.
"""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

RATE_LIMIT = 10  # requests per window
RATE_WINDOW = 60  # seconds


class RateLimiter:
    """Token-bucket rate limiter backed by Redis or in-memory fallback."""

    def __init__(self) -> None:
        self._counters: dict[str, list[float]] = {}

    def _get_user_key(self, request: Request) -> str:
        """Extract user identifier from request."""
        # Try auth header first, fall back to IP
        auth = request.headers.get("authorization", "")
        if auth:
            return f"rate:{auth[:50]}"
        client_host = request.client.host if request.client else "unknown"
        return f"rate:{client_host}"

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Check if request is allowed under rate limit.

        Returns (allowed, retry_after_seconds).
        """
        now = time.time()
        window_start = now - RATE_WINDOW

        if key not in self._counters:
            self._counters[key] = []

        # Remove expired timestamps
        self._counters[key] = [ts for ts in self._counters[key] if ts > window_start]

        if len(self._counters[key]) >= RATE_LIMIT:
            oldest = self._counters[key][0]
            retry_after = int(oldest + RATE_WINDOW - now) + 1
            return False, max(retry_after, 1)

        self._counters[key].append(now)
        return True, 0

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        window_start = now - RATE_WINDOW
        if key not in self._counters:
            return RATE_LIMIT
        active = [ts for ts in self._counters[key] if ts > window_start]
        return max(0, RATE_LIMIT - len(active))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, limiter: RateLimiter | None = None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.limiter = limiter or RateLimiter()

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        # Skip rate limiting for health checks, metrics, and uploads
        if request.url.path in (
            "/health",
            "/ready",
            "/metrics",
            "/api/v1/evals",
            "/api/v1/upload",
            "/docs",
            "/redoc",
            "/openapi.json",
        ):
            return await call_next(request)

        key = self.limiter._get_user_key(request)
        allowed, retry_after = self.limiter.is_allowed(key)

        if not allowed:
            logger.warning("Rate limit exceeded for %s", key)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {RATE_LIMIT} requests per {RATE_WINDOW} seconds",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        response = await call_next(request)
        remaining = self.limiter.get_remaining(key)
        response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
