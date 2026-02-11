import os
import hashlib
import json
import numpy as np
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
from src.rag.cache import CacheManager

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        # Lazy loading could be implemented here
        self.model = SentenceTransformer(model_name)
        self.cache = CacheManager()

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Check cache first (for single texts, or handle batch)
        # For batch, it's complex to check individual cache hits effectively without breaking batch.
        # MVP: Generate all, then cache. Or check one by one?
        # Let's generate all for efficiency using the model.
        
        # Calculate embeddings
        # model.encode returns numpy array or list
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        
        # Convert to list
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
            
        # Cache them
        for text, emb in zip(texts, embeddings):
            key = hashlib.sha256(text.encode()).hexdigest()
            await self.cache.set_embedding(key, emb)
            
        return embeddings
