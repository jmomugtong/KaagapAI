"""
Ollama LLM Client for MedQuery

Async HTTP client for the Ollama API with:
- Retry logic with exponential backoff
- Configurable timeout
- Health check endpoint
- Graceful fallback on failure
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

# Defaults from environment / docker-compose
DEFAULT_BASE_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
DEFAULT_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))

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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

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
