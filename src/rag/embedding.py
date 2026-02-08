"""
MedQuery Embedding Generation Module

Generates 384-dimensional embeddings using sentence-transformers.
Uses all-MiniLM-L6-v2 model for efficient embedding generation.
"""

import hashlib

import numpy as np

# ============================================
# Constants
# ============================================

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIMENSION = 384

# Default model name
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================
# Embedding Generator
# ============================================


class EmbeddingGenerator:
    """Generates embeddings for text using sentence-transformers.

    Uses lazy loading to defer model initialization until first use,
    improving startup time when embeddings aren't immediately needed.

    Attributes:
        model_name: Name of the sentence-transformers model to use.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Initialize the embedding generator.

        Args:
            model_name: Name of the sentence-transformers model.
                Default is "all-MiniLM-L6-v2" which produces 384-dim vectors.
        """
        self.model_name = model_name
        self._model: object | None = None

    def _get_model(self) -> object:
        """Lazily load the embedding model.

        Returns:
            Loaded SentenceTransformer model.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def generate(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to generate embedding for.

        Returns:
            384-dimensional embedding vector as list of floats.
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * EMBEDDING_DIMENSION

        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)

        # Ensure it's a 1D array and convert to list
        if isinstance(embedding, np.ndarray):
            embedding = embedding.flatten().tolist()

        return embedding

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to generate embeddings for.

        Returns:
            List of 384-dimensional embedding vectors.
        """
        if not texts:
            return []

        # Handle empty strings in batch
        non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return [[0.0] * EMBEDDING_DIMENSION for _ in texts]

        model = self._get_model()
        embeddings_array = model.encode(non_empty_texts, convert_to_numpy=True)

        # Build result list with zeros for empty texts
        result: list[list[float]] = [[0.0] * EMBEDDING_DIMENSION for _ in texts]

        for idx, emb in zip(non_empty_indices, embeddings_array, strict=False):
            if isinstance(emb, np.ndarray):
                result[idx] = emb.flatten().tolist()
            else:
                result[idx] = list(emb)

        return result

    def compute_hash(self, text: str) -> str:
        """Compute SHA256 hash of text for cache key.

        Args:
            text: Text to hash.

        Returns:
            64-character hexadecimal hash string.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================
# Convenience Functions
# ============================================


def embed_text(text: str, model_name: str = DEFAULT_MODEL_NAME) -> list[float]:
    """Convenience function to embed a single text.

    Args:
        text: Text to embed.
        model_name: Model to use for embedding.

    Returns:
        384-dimensional embedding vector.
    """
    generator = EmbeddingGenerator(model_name=model_name)
    return generator.generate(text)


def embed_texts(
    texts: list[str], model_name: str = DEFAULT_MODEL_NAME
) -> list[list[float]]:
    """Convenience function to embed multiple texts.

    Args:
        texts: Texts to embed.
        model_name: Model to use for embedding.

    Returns:
        List of 384-dimensional embedding vectors.
    """
    generator = EmbeddingGenerator(model_name=model_name)
    return generator.generate_batch(texts)
