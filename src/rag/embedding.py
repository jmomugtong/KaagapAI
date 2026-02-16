import hashlib
import logging
import os

import httpx

from src.rag.cache import CacheManager

logger = logging.getLogger(__name__)

# nomic-embed-text uses task-type prefixes for optimal retrieval.
QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "

# Ollama embed API configuration
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "768"))
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str | None = None,
        ollama_url: str | None = None,
    ):
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.ollama_url = (ollama_url or DEFAULT_OLLAMA_URL).rstrip("/")
        self.cache = CacheManager()
        self.dimension = DEFAULT_EMBEDDING_DIMENSION

    async def generate_embeddings(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        """Generate embeddings via the Ollama /api/embed endpoint.

        Args:
            texts: List of text strings to embed.
            is_query: If True, prepend search_query prefix (for retrieval queries).
                      If False, prepend search_document prefix (for documents).
        """
        prefix = QUERY_PREFIX if is_query else DOCUMENT_PREFIX
        prefixed_texts = [prefix + t for t in texts]

        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(prefixed_texts), EMBEDDING_BATCH_SIZE):
            batch = prefixed_texts[i : i + EMBEDDING_BATCH_SIZE]
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        # Cache using original text (not prefixed) as key
        for text, emb in zip(texts, all_embeddings, strict=True):
            key = hashlib.sha256(text.encode()).hexdigest()
            await self.cache.set_embedding(key, emb)

        return all_embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama /api/embed for a batch of texts."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
            response = await client.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.model_name, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]


class LangChainEmbeddingAdapter:
    """Adapter to make EmbeddingGenerator compatible with LangChain's Embeddings interface.

    Used by SemanticChunker which expects synchronous embed_documents/embed_query.
    Uses synchronous httpx calls to Ollama.
    """

    def __init__(self, generator: EmbeddingGenerator):
        self._generator = generator

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding for documents (used by SemanticChunker)."""
        prefixed = [DOCUMENT_PREFIX + t for t in texts]
        return self._sync_embed(prefixed)

    def embed_query(self, text: str) -> list[float]:
        """Synchronous embedding for a query."""
        prefixed = [QUERY_PREFIX + text]
        return self._sync_embed(prefixed)[0]

    def _sync_embed(self, texts: list[str]) -> list[list[float]]:
        """Synchronous call to Ollama /api/embed."""
        with httpx.Client(timeout=httpx.Timeout(120)) as client:
            response = client.post(
                f"{self._generator.ollama_url}/api/embed",
                json={"model": self._generator.model_name, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
