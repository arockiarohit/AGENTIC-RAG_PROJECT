"""
Embedding module.
Wraps SentenceTransformers for LangChain-compatible embeddings.
"""

from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL
import numpy as np


class EmbeddingService:
    """Singleton-style embedding service using SentenceTransformers."""

    _instance = None
    _model = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """Load and cache the embedding model."""
        if cls._model is None:
            print(f"Loading embedding model: {EMBEDDING_MODEL}...")
            cls._model = SentenceTransformer(EMBEDDING_MODEL)
            print("Embedding model loaded successfully!")
        return cls._model

    @classmethod
    def embed_texts(cls, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
        
        Returns:
            numpy array of embeddings.
        """
        model = cls.get_model()
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    @classmethod
    def embed_query(cls, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: The query text.
        
        Returns:
            numpy array of the query embedding.
        """
        model = cls.get_model()
        embedding = model.encode([query], normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)
