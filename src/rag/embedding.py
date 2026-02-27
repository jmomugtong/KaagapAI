import asyncio
import hashlib
import logging
import os
from functools import lru_cache

from src.rag.cache import CacheManager

logger = logging.getLogger(__name__)

# nomic-embed-text uses task-type prefixes for optimal retrieval.
QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "

# Model configuration
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL_HF", "nomic-ai/nomic-embed-text-v1.5"
)
DEFAULT_EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "768"))
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "16"))

# Keep Ollama config for fallback compatibility
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


@lru_cache(maxsize=1)
def _load_st_model(model_name: str):
    """Load sentence-transformers model once and cache it."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformers model: %s", model_name)

    # Try direct load first (sentence-transformers >= 2.3 forwards trust_remote_code)
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
        logger.info(
            "Model loaded — dimension: %d", model.get_sentence_embedding_dimension()
        )
        return model
    except TypeError:
        pass  # older sentence-transformers doesn't accept the kwarg

    # Fallback: load via modules so trust_remote_code reaches AutoModel directly.
    # Works with sentence-transformers 2.2.x.
    logger.info("Retrying via st_models.Transformer with trust_remote_code=True")
    from sentence_transformers import models as st_models

    word_embedding = st_models.Transformer(
        model_name,
        model_args={"trust_remote_code": True},
        tokenizer_args={"trust_remote_code": True},
        config_args={"trust_remote_code": True},
    )
    pooling = st_models.Pooling(word_embedding.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding, pooling])
    logger.info(
        "Model loaded — dimension: %d", model.get_sentence_embedding_dimension()
    )
    return model


def _encode_sync(model, texts: list[str], batch_size: int) -> list[list[float]]:
    """Run model.encode synchronously — designed to be called via to_thread.

    Processes in small batches to avoid tensor size mismatches that occur
    when variable-length texts are encoded together in large batches.
    Falls back to one-by-one encoding if a batch fails.
    """
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            batch_np = model.encode(
                batch, batch_size=batch_size, show_progress_bar=False
            )
            all_embeddings.extend(emb.tolist() for emb in batch_np)
        except RuntimeError:
            # Tensor mismatch — fall back to encoding one at a time
            for t in batch:
                try:
                    single = model.encode([t], show_progress_bar=False)
                    all_embeddings.append(single[0].tolist())
                except Exception:
                    # Zero vector as last resort so chunk still gets stored
                    dim = model.get_sentence_embedding_dimension()
                    all_embeddings.append([0.0] * dim)
    return all_embeddings


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
        self._st_model = None

    def _get_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._st_model is None:
            self._st_model = _load_st_model(self.model_name)
        return self._st_model

    async def generate_embeddings(
        self, texts: list[str], is_query: bool = False, cache: bool = True
    ) -> list[list[float]]:
        """Generate embeddings using sentence-transformers (local, fast).

        Runs the CPU-bound encode in a thread pool so it doesn't block the
        asyncio event loop.

        Args:
            texts: List of text strings to embed.
            is_query: If True, prepend search_query prefix (for retrieval queries).
                      If False, prepend search_document prefix (for documents).
            cache: If True, write each embedding to Redis. Set False for bulk
                   uploads where embeddings are persisted in Postgres directly.
        """
        prefix = QUERY_PREFIX if is_query else DOCUMENT_PREFIX
        prefixed_texts = [prefix + t for t in texts]

        model = self._get_model()
        all_embeddings = await asyncio.to_thread(
            _encode_sync, model, prefixed_texts, EMBEDDING_BATCH_SIZE
        )

        if cache:
            for text, emb in zip(texts, all_embeddings, strict=True):
                key = hashlib.sha256(text.encode()).hexdigest()
                await self.cache.set_embedding(key, emb)

        return all_embeddings


class LangChainEmbeddingAdapter:
    """Adapter to make EmbeddingGenerator compatible with LangChain's Embeddings interface.

    Used by SemanticChunker which expects synchronous embed_documents/embed_query.
    """

    def __init__(self, generator: EmbeddingGenerator):
        self._generator = generator

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding for documents."""
        prefixed = [DOCUMENT_PREFIX + t for t in texts]
        model = self._generator._get_model()
        return _encode_sync(model, prefixed, EMBEDDING_BATCH_SIZE)

    def embed_query(self, text: str) -> list[float]:
        """Synchronous embedding for a query."""
        prefixed = [QUERY_PREFIX + text]
        model = self._generator._get_model()
        return _encode_sync(model, prefixed, EMBEDDING_BATCH_SIZE)[0]
