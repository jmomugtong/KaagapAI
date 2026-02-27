"""
Ollama LLM Client for KaagapAI

Async HTTP client for the Ollama API with:
- Retry logic with exponential backoff
- Configurable timeout
- Health check endpoint
- Streaming support for low-latency first-token delivery
- Graceful fallback on failure
"""

import json
import logging
import os
from collections.abc import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

# Defaults from environment / docker-compose
DEFAULT_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
DEFAULT_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))

# LLM generation parameters
DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "200"))
DEFAULT_TOP_P = float(os.environ.get("LLM_TOP_P", "0.9"))
DEFAULT_NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "2048"))
DEFAULT_NUM_THREAD = int(os.environ.get("LLM_NUM_THREAD", "0"))
DEFAULT_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "60m")

# Stop sequences to prevent post-answer rambling
STOP_SEQUENCES = [
    "\n\n---",
    "\n\nNote:",
    "\n\nDisclaimer:",
    "\n\nSources:",
    "\n\nReferences:",
]

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


class OllamaClient:
    """Async client for Ollama LLM inference API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = DEFAULT_TOP_P,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_thread: int = DEFAULT_NUM_THREAD,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.num_ctx = num_ctx
        self.num_thread = num_thread
        self.keep_alive = keep_alive

    async def generate(self, prompt: str) -> str:
        """
        Send a prompt to Ollama and return the generated text.

        Retries up to MAX_RETRIES times with exponential backoff.
        Returns empty string on failure (graceful degradation).
        """
        import asyncio

        for attempt in range(MAX_RETRIES):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout)
                ) as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "keep_alive": self.keep_alive,
                            "options": self._build_options(),
                            "stop": STOP_SEQUENCES,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data.get("response", "")

            except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                logger.warning(
                    "Ollama timeout (attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                )
            except httpx.ConnectError as e:
                logger.warning(
                    "Ollama connection error (attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                )
            except httpx.HTTPStatusError as e:
                logger.warning(
                    "Ollama HTTP error %d (attempt %d/%d): %s",
                    e.response.status_code,
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                )
            except Exception as e:
                logger.error(
                    "Unexpected Ollama error (attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                )

            # Exponential backoff before retry (skip on last attempt)
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_BASE**attempt
                logger.info("Retrying in %ds...", wait)
                await asyncio.sleep(wait)

        logger.error("Ollama generation failed after %d attempts", MAX_RETRIES)
        return ""

    def _build_options(self) -> dict:
        """Build the Ollama options dict from instance configuration."""
        options: dict = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
            "num_ctx": self.num_ctx,
            "top_k": 10,
            "num_batch": 512,
        }
        if self.num_thread > 0:
            options["num_thread"] = self.num_thread
        return options

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream tokens from Ollama as they are generated.

        Yields individual token strings. Falls back to non-streaming
        generate() on error, yielding the full response as a single chunk.
        """
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": True,
                        "keep_alive": self.keep_alive,
                        "options": self._build_options(),
                        "stop": STOP_SEQUENCES,
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done", False):
                                return
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning("Streaming failed, falling back to non-streaming: %s", e)
            full = await self.generate(prompt)
            if full:
                yield full

    async def warmup(self) -> bool:
        """
        Pre-load the model into Ollama memory without generating a long response.

        Sends a minimal prompt with num_predict=1 so the model is loaded
        into RAM but we don't waste time generating tokens.
        Returns True on success.
        """
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": "hi",
                        "stream": False,
                        "keep_alive": self.keep_alive,
                        "options": {"num_predict": 1},
                    },
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.warning("LLM warmup failed: %s", e)
            return False

    async def health_check(self) -> bool:
        """
        Check if Ollama is reachable.

        Returns True if the Ollama API responds with 200, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
                response = await client.get(self.base_url)
                return response.status_code == 200
        except Exception as e:
            logger.warning("Ollama health check failed: %s", e)
            return False
